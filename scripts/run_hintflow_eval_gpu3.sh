#!/usr/bin/env bash
# HintFlow 128 EM on GPU3-only serve (OSS :8006 + orch :8086).
set -euo pipefail
cd /home/ivaning/PAgent

MODE=${MODE:-retained}
SESSION=${SESSION:-pagent-hintflow-${MODE}-eval-gpu3}
LOG=${LOG:-logs/eval-hintflow-${MODE}-gpu3.log}
OUT=${OUT:-checkpoints/eval_hintflow_${MODE}_gpu3_128}
mkdir -p logs "$OUT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing session $SESSION"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  python HintFlow/eval_hintflow.py \
    --data-file data/DAPO-Math.parquet \
    --limit 128 \
    --workers 4 \
    --solver-urls http://127.0.0.1:8006/v1 \
    --orch-url http://127.0.0.1:8086/v1 \
    --orch-model qwen3-4b \
    --solver-model gpt-oss-20b \
    --runtime-mode $MODE \
    --orch-temperature 0 \
    --out-dir $OUT \
    2>&1 | tee $LOG
  echo '=== DONE ===' | tee -a $LOG
"

echo "started tmux: $SESSION (GPU3-only)"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG"
echo "  out:    $OUT/summary.json"
