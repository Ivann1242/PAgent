#!/usr/bin/env bash
# Serve OSS + orch on physical GPU 3 ONLY (same card, split util).
set -euo pipefail
cd /home/ivaning/PAgent
mkdir -p logs

GPU="${GPU:-3}"
OSS_MODEL="${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}"
ORCH_MODEL="${MERGED:-checkpoints/hintflow_dpo_v2_merged}"
OSS_PORT="${OSS_PORT:-8006}"
ORCH_PORT="${ORCH_PORT:-8086}"
# 20k solver generations need prompt + completion headroom.
# Keep util low enough to co-locate Blind FF on GPU3 (other jobs may share the card).
OSS_MAX_LEN="${OSS_MAX_LEN:-32768}"
ORCH_MAX_LEN="${ORCH_MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
OSS_UTIL="${OSS_UTIL:-0.52}"
ORCH_UTIL="${ORCH_UTIL:-0.18}"

kill_port() {
  local port=$1
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
}

wait_model() {
  local port=$1 name=$2
  local i
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q "$name"; then
      echo "  :${port} OK ($name)"
      return 0
    fi
    sleep 5
  done
  echo "  :${port} FAILED" >&2
  return 1
}

echo "Stopping old servers on ${OSS_PORT}/${ORCH_PORT}..."
kill_port "$OSS_PORT"
kill_port "$ORCH_PORT"
sleep 4

echo "Starting OSS on GPU ${GPU} -> :${OSS_PORT}"
CUDA_VISIBLE_DEVICES="$GPU" nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" \
  --port "$OSS_PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$OSS_UTIL" \
  --max-model-len "$OSS_MAX_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --served-model-name gpt-oss-20b \
  > logs/oss-gpu3-${OSS_PORT}.log 2>&1 &
echo "  OSS pid=$!"
wait_model "$OSS_PORT" gpt-oss-20b

echo "Starting orch on GPU ${GPU} -> :${ORCH_PORT}"
CUDA_VISIBLE_DEVICES="$GPU" nohup python -m vllm.entrypoints.openai.api_server \
  --model "$ORCH_MODEL" \
  --port "$ORCH_PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$ORCH_UTIL" \
  --max-model-len "$ORCH_MAX_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --served-model-name qwen3-4b \
  > logs/orch-gpu3-${ORCH_PORT}.log 2>&1 &
echo "  orch pid=$!"
wait_model "$ORCH_PORT" qwen3-4b

echo "GPU${GPU}-only serve ready: OSS :${OSS_PORT} + orch :${ORCH_PORT}"
nvidia-smi -i "$GPU" --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
