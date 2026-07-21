#!/usr/bin/env bash
# Train, serve, evaluate and gate one immutable residual cycle on GPU3.
set -euo pipefail
cd /home/ivaning/PAgent

CYCLE="${CYCLE:-cycle_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-checkpoints/residual_feedback_dataset}"
ROOT="checkpoints/residual_cycles/$CYCLE"
MODEL="$ROOT/merged"
DEV_LIMIT="${DEV_LIMIT:-512}"
CANDIDATE_SUMMARY="${CANDIDATE_SUMMARY:-checkpoints/eval_residual_fixed_dev64/summary.json}"
PROMOTION_FILE=checkpoints/residual_evolution/promoted.json

PREVIOUS_MODEL=""
if [[ -f "$PROMOTION_FILE" ]]; then
  PREVIOUS_MODEL="$(python -c 'import json; print(json.load(open("checkpoints/residual_evolution/promoted.json")).get("model",""))')"
fi
if [[ -z "$PREVIOUS_MODEL" ]]; then
  PREVIOUS_MODEL=checkpoints/hintflow_dpo_v2_merged
fi

restore_previous() {
  if [[ -n "$PREVIOUS_MODEL" && -d "$PREVIOUS_MODEL" ]]; then
    MERGED="$PREVIOUS_MODEL" bash scripts/serve_residual_gpu3.sh
  fi
}
trap restore_previous ERR

CYCLE="$CYCLE" \
DATA_DIR="$DATA_DIR" \
ADAPTER_DIR="$ROOT/adapter" \
MERGED_DIR="$MODEL" \
bash scripts/train_residual_feedback_gpu3.sh

MERGED="$MODEL" bash scripts/serve_residual_gpu3.sh

MODEL_TAG="$MODEL" \
DATA_FILE=checkpoints/residual_splits/dev.jsonl \
LIMIT="$DEV_LIMIT" \
MODE=adaptive \
FEEDBACK_MODE=trained \
SELECTOR_MODE=orch \
OUT="$ROOT/eval_dev" \
LOG="$ROOT/eval_dev.log" \
bash scripts/run_residual_eval_gpu3.sh

python HintFlow/evolve_residual.py \
  --stage full \
  --cycle "$CYCLE" \
  --candidate-summary "$CANDIDATE_SUMMARY" \
  --feedback-meta "$MODEL/feedback_meta.json" \
  --eval-summary "$ROOT/eval_dev/summary.json" \
  --model "$MODEL" \
  --previous-model "$PREVIOUS_MODEL"

trap - ERR
echo "cycle complete: $CYCLE"
