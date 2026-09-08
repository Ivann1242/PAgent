#!/usr/bin/env bash
# Serve Qwen3-14B on GPUs with vLLM (OpenAI API, continuous batching).
# Drop-in replacement for gpt-oss-20b on ports 8006-8009.
#
# Reproducibility (batch-invariant + seed):
#   VLLM_BATCH_INVARIANT=1  → output independent of batch size / schedule
#   --seed 0 on every engine
#   clients should use temperature=0 and pass seed=
# See https://docs.vllm.ai/en/stable/features/batch_invariance/
#
# Default layout: GPU0-3 -> :8006-8009 (same as old OSS).
# If :8086 Blind FF router is on GPU0, set SKIP_GPU0=1 to only start 8007-8009.
set -euo pipefail
cd /home/ivaning/PAgent

MODEL="${SOLVER_MODEL_PATH:-${OSS_MODEL_PATH:-/home/ivaning/models/Qwen3-14B}}"
SERVED_NAME="${SOLVER_SERVED_NAME:-qwen3-14b}"
UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_LEN="${SOLVER_MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${SOLVER_MAX_NUM_SEQS:-32}"
SEED="${SOLVER_SEED:-0}"
BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
LOG_DIR="${LOG_DIR:-/home/ivaning/PAgent/logs}"
mkdir -p "$LOG_DIR"

if [[ ! -d "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Download first: hf download Qwen/Qwen3-14B --local-dir $MODEL" >&2
  exit 1
fi

declare -a PAIRS=(
  "0:8006"
  "1:8007"
  "2:8008"
  "3:8009"
)
if [[ "${SKIP_GPU0:-0}" == "1" ]]; then
  PAIRS=("1:8007" "2:8008" "3:8009")
fi

stop_port() {
  local port=$1
  # Use pgrep|xargs so this script's own cmdline is not matched by pkill -f.
  local pids
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}" || true)
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
  fi
  pids=$(pgrep -f "serve_hf_deterministic.py.*--port ${port}" || true)
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
  fi
}

echo "Stopping old solvers on target ports..."
for pair in "${PAIRS[@]}"; do
  stop_port "${pair##*:}"
done
sleep 5

for pair in "${PAIRS[@]}"; do
  gpu="${pair%%:*}"
  port="${pair##*:}"
  # GPU0 may share with Blind FF router
  util="$UTIL"
  seqs="$MAX_NUM_SEQS"
  if [[ "$gpu" == "0" && "${SHARE_GPU0:-0}" == "1" ]]; then
    util="${GPU0_MEM_UTIL:-0.78}"
    seqs="${GPU0_MAX_NUM_SEQS:-16}"
  fi
  echo "Starting ${SERVED_NAME} on GPU${gpu} -> :${port} (util=${util}, seqs=${seqs}, seed=${SEED}, batch_invariant=${BATCH_INVARIANT})"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_BATCH_INVARIANT="$BATCH_INVARIANT" \
    nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$port" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$util" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$seqs" \
    --seed "$SEED" \
    --served-model-name "$SERVED_NAME" \
    > "${LOG_DIR}/qwen3-14b-${port}.log" 2>&1 &
  echo "  pid=$! log=${LOG_DIR}/qwen3-14b-${port}.log"
done

echo "Waiting for endpoints..."
fail=0
for pair in "${PAIRS[@]}"; do
  port="${pair##*:}"
  ok=0
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q "$SERVED_NAME"; then
      echo "  :${port} OK"
      ok=1
      break
    fi
    sleep 5
  done
  if [[ "$ok" -eq 0 ]]; then
    echo "  :${port} FAILED — see ${LOG_DIR}/qwen3-14b-${port}.log" >&2
    fail=1
  fi
done

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "Qwen3-14B ready (vLLM batch max_num_seqs=${MAX_NUM_SEQS}, seed=${SEED}): ports ${PAIRS[*]}"
