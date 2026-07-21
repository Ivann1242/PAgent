#!/usr/bin/env bash
# Small synchronous collect→export→train→serve→adaptive-eval smoke on GPU3 only.
set -euo pipefail
cd /home/ivaning/PAgent

GPU=3
CYCLE="${CYCLE:-smoke_$(date +%Y%m%d_%H%M%S)}"
ROOT="checkpoints/residual_cycles/$CYCLE"
TRAIN_LIMIT="${TRAIN_LIMIT:-64}"
DEV_LIMIT="${DEV_LIMIT:-32}"
EVAL_LIMIT="${EVAL_LIMIT:-16}"

python HintFlow/make_residual_splits.py --dev-size 512

python HintFlow/collect_residual_feedback.py \
  --data-file checkpoints/residual_splits/train.jsonl \
  --limit "$TRAIN_LIMIT" \
  --workers 4 \
  --samples-per-action 1 \
  --branch-only-if-baseline-wrong \
  --branch-max-tokens 2048 \
  --out "$ROOT/train_turns.jsonl"

python HintFlow/collect_residual_feedback.py \
  --data-file checkpoints/residual_splits/dev.jsonl \
  --limit "$DEV_LIMIT" \
  --workers 4 \
  --samples-per-action 1 \
  --branch-only-if-baseline-wrong \
  --branch-max-tokens 2048 \
  --out "$ROOT/dev_turns.jsonl"

python HintFlow/export_residual_feedback.py \
  --counterfactual "$ROOT/train_turns.jsonl,$ROOT/dev_turns.jsonl" \
  --out-dir "$ROOT/dataset"

CYCLE="$CYCLE" \
DATA_DIR="$ROOT/dataset" \
ADAPTER_DIR="$ROOT/adapter" \
MERGED_DIR="$ROOT/merged" \
EPOCHS=1 \
MAX_PER_TASK=32 \
bash scripts/train_residual_feedback_gpu3.sh

MERGED="$ROOT/merged" bash scripts/serve_residual_gpu3.sh

MODEL_TAG="$ROOT/merged" \
DATA_FILE=checkpoints/residual_splits/dev.jsonl \
LIMIT="$EVAL_LIMIT" \
MODE=adaptive \
FEEDBACK_MODE=trained \
SELECTOR_MODE=orch \
OUT="$ROOT/eval_dev" \
LOG="$ROOT/eval_dev.log" \
bash scripts/run_residual_eval_gpu3.sh

echo "smoke cycle complete: $ROOT"
