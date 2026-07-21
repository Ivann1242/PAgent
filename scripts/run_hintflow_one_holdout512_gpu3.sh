#!/usr/bin/env bash
# HintFlow_one on leakage-free DAPO holdout-512, GPU3 serves only.
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=${SESSION:-pagent-hintflow-one-holdout512-gpu3}
LOG=${LOG:-logs/eval-hintflow-one-holdout512-gpu3.log}
OUT=${OUT:-checkpoints/eval_hintflow_one_holdout512_20k}
DATA=${DATA:-data/dapo_holdout_512.jsonl}
ORCH_MODEL=${ORCH_MODEL:-qwen3-4b-blind-ff-17k}
TOKENS=${TOKENS:-20000}
WORKERS=${WORKERS:-1}
SEED=${SEED:-41}

mkdir -p logs "$OUT"

if [[ ! -f "$DATA" ]]; then
  echo "building holdout: $DATA"
  python scripts/build_dapo_holdout.py --out-file "$DATA" --n 512
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing session $SESSION"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  echo '=== '\$(date -Is)' HintFlow_one holdout512 ===' | tee $LOG
  curl -sf http://127.0.0.1:8006/v1/models | grep -q gpt-oss-20b
  curl -sf http://127.0.0.1:8086/v1/models | grep -q $ORCH_MODEL
  python HintFlow_one/eval_one.py \\
    --data-file $DATA \\
    --limit 512 \\
    --workers $WORKERS \\
    --solver-urls http://127.0.0.1:8006/v1 \\
    --orch-url http://127.0.0.1:8086/v1 \\
    --orch-model $ORCH_MODEL \\
    --solver-model gpt-oss-20b \\
    --solver-max-tokens $TOKENS \\
    --selector-mode orch \\
    --replace-threshold 0.90 \\
    --seed $SEED \\
    --out-dir $OUT \\
    2>&1 | tee -a $LOG
  echo '=== DONE ===' | tee -a $LOG
  cat $OUT/summary.json | tee -a $LOG
"

echo "started tmux: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG"
echo "  out:    $OUT/summary.json"
echo "  data:   $DATA (+ ${DATA%.jsonl}.stats.json)"
