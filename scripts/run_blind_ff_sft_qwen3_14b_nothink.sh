#!/usr/bin/env bash
# SFT free-form prompt optimizer on Qwen3-14B-nothink blind flip labels.
# Same training recipe as before (epochs/bs/lr); engineering: torchrun DDP on 4 GPUs
# with global batch_size unchanged (micro_bs = batch_size / nproc).
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=${SESSION:-pagent-blind-ff-sft-qwen3-14b-nothink}
LABELS=${LABELS:-checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels.jsonl}
ADAPTER=${ADAPTER:-checkpoints/blind_ff_sft_qwen3_14b_nothink_adapter}
MERGED=${MERGED:-checkpoints/blind_ff_sft_qwen3_14b_nothink_merged}
LOG_FILE=${LOG_FILE:-$LOG_DIR/blind-ff-sft-qwen3-14b-nothink-train.log}
GPU=${GPU:-0,1,2,3}
NPROC=${NPROC:-4}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-4}
LR=${LR:-2e-5}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

if [[ ! -f "$LABELS" ]]; then
  echo "missing labels: $LABELS" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$GPU"
if [[ "${#GPU_ARR[@]}" -ne "$NPROC" ]]; then
  echo "GPU list ($GPU) length must equal NPROC=$NPROC" >&2
  exit 1
fi

echo "GPUs=$GPU nproc=$NPROC (global_bs=$BATCH => micro_bs=$((BATCH / NPROC)))"

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES=$GPU

  echo \"=== \$(date -Is) ff-sft-train (qwen3-14b-nothink labels, DDP) ===\" | tee '$LOG_FILE'
  echo \"labels=$LABELS n=\$(wc -l < $LABELS) gpus=$GPU nproc=$NPROC epochs=$EPOCHS global_bs=$BATCH lr=$LR\" | tee -a '$LOG_FILE'

  torchrun --standalone --nproc_per_node=$NPROC run.py ff-sft-train \
    --labels-file $LABELS \
    --out-dir $ADAPTER \
    --gpu $GPU \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    --lr $LR \
    2>&1 | tee -a '$LOG_FILE'

  echo \"=== \$(date -Is) ff-sft-merge ===\" | tee -a '$LOG_FILE'
  python run.py ff-sft-merge \
    --adapter-dir $ADAPTER \
    --merged-dir $MERGED \
    2>&1 | tee -a '$LOG_FILE'

  echo \"=== \$(date -Is) DONE ===\" | tee -a '$LOG_FILE'
  cat $ADAPTER/sft_meta.json | tee -a '$LOG_FILE'
"

echo "started tmux: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG_FILE"
echo "  labels: $(wc -l < "$LABELS") rows from $LABELS"
echo "  gpus:   CUDA_VISIBLE_DEVICES=$GPU (torchrun nproc=$NPROC)"
echo "  recipe: epochs=$EPOCHS global_bs=$BATCH lr=$LR (unchanged)"
echo "  out:    $ADAPTER -> $MERGED"
