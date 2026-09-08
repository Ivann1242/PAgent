#!/usr/bin/env bash
# SFT on cleaned Qwen3-14B-nothink flip labels (true baseline=0 & hint=1).
# Same recipe: epochs=3, global_bs=4, lr=2e-5; torchrun DDP 4 GPUs.
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=${SESSION:-pagent-blind-ff-sft-qwen3-14b-nothink-clean}
LABELS=${LABELS:-checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_clean.jsonl}
ADAPTER=${ADAPTER:-checkpoints/blind_ff_sft_qwen3_14b_nothink_clean_adapter}
MERGED=${MERGED:-checkpoints/blind_ff_sft_qwen3_14b_nothink_clean_merged}
LOG_FILE=${LOG_FILE:-$LOG_DIR/blind-ff-sft-qwen3-14b-nothink-clean-train.log}
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

echo "GPUs=$GPU nproc=$NPROC labels=$(wc -l < "$LABELS") -> $ADAPTER"

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES=$GPU

  echo \"=== \$(date -Is) ff-sft-train CLEAN labels DDP ===\" | tee '$LOG_FILE'
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
echo "  log: tail -f $LOG_FILE"
echo "  out: $ADAPTER -> $MERGED"
