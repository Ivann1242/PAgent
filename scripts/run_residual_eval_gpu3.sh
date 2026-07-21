#!/usr/bin/env bash
# Evaluate residual feedback policy using physical GPU3 endpoints only.
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=pagent-residual-eval-gpu3
LOG="${LOG:-logs/eval-residual-gpu3.log}"
OUT="${OUT:-checkpoints/eval_residual_128}"
MODE="${MODE:-adaptive}"
FEEDBACK_MODE="${FEEDBACK_MODE:-trained}"
SELECTOR_MODE="${SELECTOR_MODE:-orch}"
MODEL_TAG="${MODEL_TAG:-}"
WORKERS="${WORKERS:-4}"
DETACH="${DETACH:-0}"
FINAL_EVAL="${FINAL_EVAL:-0}"
if [[ "$FINAL_EVAL" == "1" ]]; then
  DATA_FILE="${DATA_FILE:-checkpoints/residual_splits/final.jsonl}"
  LIMIT="${LIMIT:-128}"
else
  DATA_FILE="${DATA_FILE:-checkpoints/residual_splits/dev.jsonl}"
  LIMIT="${LIMIT:-512}"
fi
mkdir -p logs "$OUT"

EVAL_CMD=(
  python HintFlow/eval_residual.py
  --data-file "$DATA_FILE"
  --limit "$LIMIT"
  --workers "$WORKERS"
  --solver-urls http://127.0.0.1:8006/v1
  --orch-url http://127.0.0.1:8086/v1
  --orch-model qwen3-4b
  --solver-model gpt-oss-20b
  --model-tag "$MODEL_TAG"
  --max-solver-calls 7
  --policy-mode "$MODE"
  --selector-mode "$SELECTOR_MODE"
  --feedback-mode "$FEEDBACK_MODE"
  --out-dir "$OUT"
)

if [[ "$DETACH" == "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
  fi
  quoted="$(printf '%q ' "${EVAL_CMD[@]}")"
  tmux new-session -d -s "$SESSION" bash -lc \
    "cd /home/ivaning/PAgent && CUDA_VISIBLE_DEVICES=3 $quoted 2>&1 | tee '$LOG'"
  echo "started detached: $SESSION (physical GPU3 only)"
else
  CUDA_VISIBLE_DEVICES=3 "${EVAL_CMD[@]}" 2>&1 | tee "$LOG"
fi
echo "summary: $OUT/summary.json"
