#!/usr/bin/env bash
# Restart gpt-oss-20b on 3 GPUs with ~35GB VRAM each (free room for other jobs).
set -euo pipefail

MODEL="${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}"
UTIL="${GPU_MEM_UTIL:-0.36}"          # ~35GB on 98GB cards
MAX_LEN="${OSS_MAX_MODEL_LEN:-8192}"
LOG_DIR="${LOG_DIR:-/home/ivaning/PAgent/logs}"
mkdir -p "$LOG_DIR"

# GPU1/8007 often occupied by others; only (re)start these three.
declare -a GPUS=(0 2 3)
declare -a PORTS=(8006 8008 8009)

stop_one() {
  local port=$1
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}.*gpt-oss-20b" 2>/dev/null || true
}

start_one() {
  local gpu=$1 port=$2
  echo "Starting OSS-20B compact on GPU ${gpu} -> :${port} (util=${UTIL}, max_len=${MAX_LEN})"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$port" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAX_LEN" \
    --served-model-name gpt-oss-20b \
    > "${LOG_DIR}/oss-${port}.log" 2>&1 &
  echo "  pid=$! log=${LOG_DIR}/oss-${port}.log"
}

echo "Stopping existing OSS on ports ${PORTS[*]}..."
for port in "${PORTS[@]}"; do
  stop_one "$port"
done
sleep 8

for i in "${!GPUS[@]}"; do
  start_one "${GPUS[$i]}" "${PORTS[$i]}"
done

echo "Waiting for endpoints..."
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
    exit 1
  fi
done

echo "Compact OSS ready on ${PORTS[*]} (~${UTIL} GPU util, max_model_len=${MAX_LEN})"
