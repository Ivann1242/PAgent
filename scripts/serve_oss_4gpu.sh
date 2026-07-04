#!/usr/bin/env bash
# Deploy gpt-oss-20b on 4 GPUs (one vLLM per GPU) for parallel labeling.
# Stop router/qwen/other GPU jobs before running if VRAM is tight.
set -euo pipefail

MODEL="${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}"
UTIL="${GPU_MEM_UTIL:-0.70}"
LOG_DIR="${LOG_DIR:-/home/ivaning/PAgent/logs}"
mkdir -p "$LOG_DIR"

declare -a GPUS=(0 1 2 3)
declare -a PORTS=(8006 8007 8008 8009)

start_one() {
  local gpu=$1 port=$2
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "GPU ${gpu} port ${port}: already up, skip"
    return
  fi
  echo "Starting OSS-20B on GPU ${gpu} -> :${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$port" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$UTIL" \
    --served-model-name gpt-oss-20b \
    > "${LOG_DIR}/oss-${port}.log" 2>&1 &
  echo "  pid=$! log=${LOG_DIR}/oss-${port}.log"
}

for i in "${!GPUS[@]}"; do
  start_one "${GPUS[$i]}" "${PORTS[$i]}"
done

echo "Waiting for endpoints..."
for port in "${PORTS[@]}"; do
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q gpt-oss-20b; then
      echo "  :${port} OK"
      break
    fi
    sleep 5
  done
done

echo "Ready: ${PORTS[*]}"
