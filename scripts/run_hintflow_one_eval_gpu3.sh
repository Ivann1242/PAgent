#!/usr/bin/env bash
# HintFlow_one (Blind FF + selector) 128 eval on GPU3 serve.
set -euo pipefail
cd /home/ivaning/PAgent

MODE=${MODE:-orch}
SESSION=${SESSION:-pagent-hintflow-one-${MODE}-eval-gpu3}
LOG=${LOG:-logs/eval-hintflow-one-${MODE}-gpu3.log}
OUT=${OUT:-checkpoints/eval_hintflow_one_${MODE}_gpu3_128}
SEED=${SEED:-41}
TOKENS=${TOKENS:-4096}

mkdir -p logs "$OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing session $SESSION"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  python HintFlow_one/eval_one.py \
    --data-file data/DAPO-Math.parquet \
    --limit 128 \
    --workers 4 \
    --solver-urls http://127.0.0.1:8006/v1 \
    --orch-url http://127.0.0.1:8086/v1 \
    --orch-model qwen3-4b \
    --solver-model gpt-oss-20b \
    --solver-max-tokens $TOKENS \
    --selector-mode $MODE \
    --replace-threshold 0.90 \
    --seed $SEED \
    --out-dir $OUT \
    2>&1 | tee $LOG
  echo '=== DONE ===' | tee -a $LOG
"

echo "started tmux: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG"
echo "  out:    $OUT/summary.json"
