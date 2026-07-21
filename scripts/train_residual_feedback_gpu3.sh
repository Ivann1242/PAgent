#!/usr/bin/env bash
# Train/merge residual feedback LoRA on physical GPU3 only, then restart serving.
set -euo pipefail
cd /home/ivaning/PAgent

DATA_DIR="${DATA_DIR:-checkpoints/residual_feedback_dataset}"
CYCLE="${CYCLE:-cycle_$(date +%Y%m%d_%H%M%S)}"
ADAPTER_DIR="${ADAPTER_DIR:-checkpoints/residual_cycles/$CYCLE/adapter}"
MERGED_DIR="${MERGED_DIR:-checkpoints/residual_cycles/$CYCLE/merged}"
EPOCHS="${EPOCHS:-2}"
MAX_PER_TASK="${MAX_PER_TASK:-0}"
BALANCED_TASK_SIZE="${BALANCED_TASK_SIZE:-1024}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
EVAL_MAX_PER_TASK="${EVAL_MAX_PER_TASK:-64}"
GENERATION_SMOKE_PER_TASK="${GENERATION_SMOKE_PER_TASK:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"

if [[ ! -s "$DATA_DIR/train.jsonl" || ! -s "$DATA_DIR/val.jsonl" ]]; then
  echo "missing residual feedback dataset in $DATA_DIR" >&2
  exit 1
fi

# Stop only our two API ports before loading the training model.
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8006' 2>/dev/null || true
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8086' 2>/dev/null || true
sleep 5

CUDA_VISIBLE_DEVICES=3 python HintFlow/train_residual_feedback.py \
  --train-file "$DATA_DIR/train.jsonl" \
  --val-file "$DATA_DIR/val.jsonl" \
  --adapter-dir "$ADAPTER_DIR" \
  --merged-dir "$MERGED_DIR" \
  --gpu 3 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --max-per-task "$MAX_PER_TASK" \
  --balanced-task-size "$BALANCED_TASK_SIZE" \
  --max-length "$MAX_LENGTH" \
  --eval-max-per-task "$EVAL_MAX_PER_TASK" \
  --generation-smoke-per-task "$GENERATION_SMOKE_PER_TASK"

echo "candidate merged model: $MERGED_DIR"
