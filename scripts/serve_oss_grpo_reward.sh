#!/usr/bin/env bash
# Start high-throughput OSS reward servers for Blind FF-GRPO.
# Default: GPU1->:8007, GPU2->:8008, GPU3->:8006 (free Blind FF on GPU3 first).
# Leaves GPU0 free for Qwen LoRA policy.
set -euo pipefail
cd /home/ivaning/PAgent

MODEL="${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}"
UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_LEN="${OSS_MAX_MODEL_LEN:-16384}"   # enough for ~8k gen + prompt
MAX_NUM_SEQS="${OSS_MAX_NUM_SEQS:-8}"
LOG_DIR="${LOG_DIR:-/home/ivaning/PAgent/logs}"
mkdir -p "$LOG_DIR"

# gpu:port
declare -a PAIRS=(
  "1:8007"
  "2:8008"
  "3:8006"
)

stop_port() {
  local port=$1
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
}

# Free GPU3 Blind FF router if present (GRPO does not need it).
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8086' 2>/dev/null || true

echo "Stopping old OSS on target ports..."
for pair in "${PAIRS[@]}"; do
  stop_port "${pair##*:}"
done
sleep 6

for pair in "${PAIRS[@]}"; do
  gpu="${pair%%:*}"
  port="${pair##*:}"
  echo "Starting OSS on GPU${gpu} -> :${port} (util=${UTIL}, max_len=${MAX_LEN}, seqs=${MAX_NUM_SEQS})"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$port" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --served-model-name gpt-oss-20b \
    > "${LOG_DIR}/oss-grpo-${port}.log" 2>&1 &
  echo "  pid=$! log=${LOG_DIR}/oss-grpo-${port}.log"
done

echo "Waiting for endpoints..."
fail=0
for pair in "${PAIRS[@]}"; do
  port="${pair##*:}"
  ok=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q gpt-oss-20b; then
      echo "  :${port} OK"
      ok=1
      break
    fi
    sleep 5
  done
  if [[ "$ok" -eq 0 ]]; then
    echo "  :${port} FAILED — see ${LOG_DIR}/oss-grpo-${port}.log" >&2
    fail=1
  fi
done

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "GRPO reward OSS ready: 8007,8008,8006 (max_num_seqs=${MAX_NUM_SEQS})"
