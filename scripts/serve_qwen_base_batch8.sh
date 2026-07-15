#!/usr/bin/env bash
# Serve base (unfinetuned) Qwen3-4B with max_num_seqs=8.
set -euo pipefail
cd /home/ivaning/PAgent

MODEL="${MODEL:-/home/ivaning/prompt-r1r/Prompt-R1/Qwen/Qwen3-4B}"
MODEL_NAME="${MODEL_NAME:-qwen3-4b}"
PORT="${PORT:-8086}"
GPU="${GPU:-3}"
UTIL="${GPU_MEM_UTIL:-0.20}"
MAX_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"
SERVE_LOG="$LOG_DIR/qwen-base-serve.log"

pkill -f "api_server.*--port ${PORT}" 2>/dev/null || true
sleep 5

echo "Starting BASE ${MODEL_NAME} on GPU ${GPU} -> :${PORT} (util=${UTIL}, max_num_seqs=${MAX_NUM_SEQS})"
echo "  model: $MODEL"
CUDA_VISIBLE_DEVICES="$GPU" nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --served-model-name "$MODEL_NAME" \
  > "$SERVE_LOG" 2>&1 &
echo "  pid=$! log=$SERVE_LOG"

ok=0
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" | grep -q "$MODEL_NAME"; then
    echo "serve ready: $MODEL_NAME (base)"
    ok=1
    break
  fi
  sleep 5
done
if [[ "$ok" -eq 0 ]]; then
  echo "FAILED — see $SERVE_LOG" >&2
  tail -40 "$SERVE_LOG" >&2 || true
  exit 1
fi

curl -sf "http://127.0.0.1:${PORT}/v1/models"
echo
nvidia-smi -i "$GPU" --query-gpu=index,memory.used,memory.free --format=csv
