#!/usr/bin/env bash
# Serve blind_ff_sft_17k_merged and run 128x4 repeat eval.
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=pagent-blind-ff-sft-17k-eval
MERGED=checkpoints/blind_ff_sft_17k_merged
MODEL_NAME=qwen3-4b-blind-ff-17k
PORT=8086
GPU=1
LOG_FILE="$LOG_DIR/eval-blind-ff-sft-17k-128-repeat.log"
SERVE_LOG="$LOG_DIR/blind-ff-sft-17k-serve.log"
OUT_ROOT=checkpoints/eval_blind_ff_sft_17k_128_repeat

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

echo "=== stop old router on :$PORT ==="
pkill -f "api_server.*--port ${PORT}" 2>/dev/null || true
pkill -f "vllm serve.*${PORT}" 2>/dev/null || true
sleep 5

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent

  echo '=== serve $MODEL_NAME on GPU $GPU :$PORT ===' | tee '$LOG_FILE'
  CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --model $MERGED \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.25 \
    --max-model-len 8192 \
    --served-model-name $MODEL_NAME \
    > '$SERVE_LOG' 2>&1 &
  SERVE_PID=\$!

  for i in \$(seq 1 60); do
    if curl -sf http://127.0.0.1:${PORT}/v1/models | grep -q '$MODEL_NAME'; then
      echo 'serve ready' | tee -a '$LOG_FILE'
      break
    fi
    sleep 5
  done
  curl -sf http://127.0.0.1:${PORT}/v1/models | tee -a '$LOG_FILE' || { echo 'serve failed'; tail -30 '$SERVE_LOG'; exit 1; }

  echo '=== eval 128 x 4 repeat ===' | tee -a '$LOG_FILE'
  python eval_repeat.py \
    --repeats 4 \
    --limit 128 \
    --router-mode ff_router \
    --router-url http://127.0.0.1:${PORT}/v1 \
    --router-model $MODEL_NAME \
    --out-root $OUT_ROOT \
    --eval-workers 32 \
    2>&1 | tee -a '$LOG_FILE'

  echo '=== DONE ===' | tee -a '$LOG_FILE'
  cat $OUT_ROOT/aggregate.json | tee -a '$LOG_FILE'
  kill \$SERVE_PID 2>/dev/null || true
"

echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG_FILE"
echo "  out:    $OUT_ROOT/aggregate.json"
