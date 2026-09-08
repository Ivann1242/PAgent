#!/usr/bin/env bash
# Deterministic HF Transformers OSS servers (OpenAI-compatible).
# Drop-in for scripts/serve_oss_grpo_reward.sh when you want lower reward/SFT noise.
#
# Layout (same ports as GRPO reward vLLM):
#   GPU1 -> :8007, GPU2 -> :8008, GPU3 -> :8006
# Leaves GPU0 free for Qwen LoRA policy.
#
# Tradeoff: ~batch=1 greedy + lock → much slower than vLLM, but stable labels.
set -euo pipefail
cd /home/ivaning/PAgent

MODEL="${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}"
SERVED_NAME="${OSS_SERVED_NAME:-gpt-oss-20b}"
DTYPE="${OSS_DTYPE:-bfloat16}"
LOG_DIR="${LOG_DIR:-/home/ivaning/PAgent/logs}"
mkdir -p "$LOG_DIR"

declare -a PAIRS=(
  "1:8007"
  "2:8008"
  "3:8006"
)

stop_port() {
  local port=$1
  pkill -f "serve_hf_deterministic.py.*--port ${port}" 2>/dev/null || true
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
}

# Free Blind FF router if present (GRPO reward path usually does not need it).
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8086' 2>/dev/null || true
pkill -f 'serve_hf_deterministic.py.*--port 8086' 2>/dev/null || true

echo "Stopping old OSS on target ports..."
for pair in "${PAIRS[@]}"; do
  stop_port "${pair##*:}"
done
sleep 6

for pair in "${PAIRS[@]}"; do
  gpu="${pair%%:*}"
  port="${pair##*:}"
  echo "Starting HF-deterministic OSS on GPU${gpu} -> :${port}"
  CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    nohup python scripts/serve_hf_deterministic.py \
      --model "$MODEL" \
      --served-model-name "$SERVED_NAME" \
      --host 127.0.0.1 \
      --port "$port" \
      --dtype "$DTYPE" \
      > "${LOG_DIR}/oss-hf-det-${port}.log" 2>&1 &
  echo "  pid=$! log=${LOG_DIR}/oss-hf-det-${port}.log"
done

echo "Waiting for endpoints (HF load can take several minutes)..."
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
    echo "  :${port} FAILED — see ${LOG_DIR}/oss-hf-det-${port}.log" >&2
    fail=1
  fi
done

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "HF-deterministic OSS ready: 8007,8008,8006"
echo "GRPO/SFT reward URLs unchanged: http://127.0.0.1:{8006,8007,8008}/v1"
echo "Use temperature=0 (greedy) clients for stable labels."
