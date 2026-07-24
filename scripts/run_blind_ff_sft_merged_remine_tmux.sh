#!/usr/bin/env bash
# SFT free-form optimizer from merged labels (17k + hard-remine), base Qwen3-4B.
# Keeps existing HF1 (blind_ff_sft_17k_*) untouched; writes new dirs.
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=pagent-blind-ff-sft-merged-remine
LABELS=checkpoints/blind_hint_merged_17k_plus_hardremine_8k/oracle_labels.jsonl
ADAPTER=checkpoints/blind_ff_sft_merged_remine_adapter
MERGED=checkpoints/blind_ff_sft_merged_remine_merged
LOG_FILE="$LOG_DIR/blind-ff-sft-merged-remine-train.log"
GPU=${GPU:-0}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

if [[ ! -f "$LABELS" ]]; then
  echo "missing labels: $LABELS"
  exit 1
fi

# Refuse to clobber current HF1 paths
for p in checkpoints/blind_ff_sft_17k_adapter checkpoints/blind_ff_sft_17k_merged; do
  if [[ "$ADAPTER" == "$p" || "$MERGED" == "$p" ]]; then
    echo "refusing to overwrite HF1 path $p"; exit 1
  fi
done

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo '=== ff-sft-train (merged 17k+hardremine, base Qwen3-4B) ===' | tee '$LOG_FILE'
  echo \"labels=\$(wc -l < $LABELS) gpu=$GPU\" | tee -a '$LOG_FILE'
  df -h /home/ivaning | tee -a '$LOG_FILE'

  python run.py ff-sft-train \
    --labels-file $LABELS \
    --out-dir $ADAPTER \
    --gpu $GPU \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-5 \
    2>&1 | tee -a '$LOG_FILE'

  echo '=== ff-sft-merge ===' | tee -a '$LOG_FILE'
  python run.py ff-sft-merge \
    --adapter-dir $ADAPTER \
    --merged-dir $MERGED \
    2>&1 | tee -a '$LOG_FILE'

  echo '=== DONE ===' | tee -a '$LOG_FILE'
  cat $ADAPTER/sft_meta.json | tee -a '$LOG_FILE'
"

echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG_FILE"
echo "  labels: $(wc -l < "$LABELS") rows from $LABELS"
echo "  gpu:    CUDA_VISIBLE_DEVICES=$GPU"
echo "  out:    $ADAPTER -> $MERGED"
echo "  HF1 kept: checkpoints/blind_ff_sft_17k_merged (:8086)"
