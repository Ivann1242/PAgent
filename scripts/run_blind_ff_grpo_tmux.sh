#!/usr/bin/env bash
# GRPO fine-tune from blind v1 SFT (no empty hints).
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=pagent-blind-ff-grpo

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

pkill -f 'run.py ff-train.*blind_ff_grpo' 2>/dev/null || true
pkill -f 'vllm serve checkpoints/blind_ff_sft_v3_merged' 2>/dev/null || true
# free GPU1 for Qwen: stop OSS on :8007 (shares GPU1 with --gpu 1)
pkill -f 'vllm.*--port 8007' 2>/dev/null || true
pkill -f 'api_server.*8007' 2>/dev/null || true
sleep 5

tmux new-session -d -s "$SESSION" bash -lc "
  cd /home/ivaning/PAgent
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python run.py ff-train \
    --init-adapter checkpoints/blind_ff_sft_adapter \
    --out-dir checkpoints/blind_ff_grpo_adapter \
    --rollout-log checkpoints/blind_ff_grpo_rollouts.jsonl \
    --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1 \
    --batch-size 32 \
    --max-steps 32 \
    --k 8 \
    --gen-batch-size 4 \
    --gpu 1 \
    --rollout-workers 32 \
    --lr 1e-6 \
    2>&1 | tee -a $LOG_DIR/blind-ff-grpo-train.log
"

echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG_DIR/blind-ff-grpo-train.log"
echo "  rollouts: wc -l checkpoints/blind_ff_grpo_rollouts.jsonl"
echo "  note: OSS :8007 stopped to free GPU1 for Qwen GRPO"
