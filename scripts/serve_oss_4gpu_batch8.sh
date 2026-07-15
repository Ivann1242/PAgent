#!/usr/bin/env bash
# Deploy gpt-oss-20b on 4 GPUs with max_num_seqs=8 (throughput-oriented, not max-batch).
set -euo pipefail

MODEL="${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}"
UTIL="${GPU_MEM_UTIL:-0.40}"
MAX_LEN="${OSS_MAX_MODEL_LEN:-50000}"
MAX_NUM_SEQS="${OSS_MAX_NUM_SEQS:-8}"
LOG_DIR="${LOG_DIR:-/home/ivaning/PAgent/logs}"
mkdir -p "$LOG_DIR"

declare -a GPUS=(0 1 2 3)
declare -a PORTS=(8006 8007 8008 8009)

stop_one() {
  local port=$1
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
}

start_one() {
  local gpu=$1 port=$2
  echo "Starting OSS-20B on GPU ${gpu} -> :${port} (util=${UTIL}, max_len=${MAX_LEN}, max_num_seqs=${MAX_NUM_SEQS})"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$port" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --served-model-name gpt-oss-20b \
    > "${LOG_DIR}/oss-${port}.log" 2>&1 &
  echo "  pid=$! log=${LOG_DIR}/oss-${port}.log"
}

echo "Stopping any existing OSS on ${PORTS[*]}..."
for port in "${PORTS[@]}"; do
  stop_one "$port"
done
sleep 8

for i in "${!GPUS[@]}"; do
  start_one "${GPUS[$i]}" "${PORTS[$i]}"
done

echo "Waiting for endpoints..."
fail=0
for port in "${PORTS[@]}"; do
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
    echo "  :${port} FAILED — see ${LOG_DIR}/oss-${port}.log" >&2
    fail=1
  fi
done

echo "=== GPU memory after start ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "OSS ready on ${PORTS[*]} (max_num_seqs=${MAX_NUM_SEQS}, util=${UTIL})"
