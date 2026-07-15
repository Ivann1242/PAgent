#!/usr/bin/env bash
# HintFlow 128-eval: 4 OSS endpoints × batch 8 → 32 workers.
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=pagent-hintflow-eval128
LOG=logs/eval-hintflow-128.log
OUT=checkpoints/eval_hintflow_128
mkdir -p logs "$OUT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing session $SESSION"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  python HintFlow/eval_hintflow.py \
    --limit 128 \
    --workers 32 \
    --solver-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1 \
    --orch-url http://127.0.0.1:8086/v1 \
    --out-dir $OUT \
    2>&1 | tee $LOG
  echo '=== DONE ===' | tee -a $LOG
"

echo "started tmux: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG"
echo "  out:    $OUT/summary.json"
