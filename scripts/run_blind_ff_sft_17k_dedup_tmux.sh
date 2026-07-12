#!/usr/bin/env bash
# SFT from deduplicated 1809 blind flip labels (base Qwen3-4B).
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=pagent-blind-ff-sft-17k-dedup
LABELS=checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl
ADAPTER=checkpoints/blind_ff_sft_17k_dedup_adapter
MERGED=checkpoints/blind_ff_sft_17k_dedup_merged
LOG_FILE="$LOG_DIR/blind-ff-sft-17k-dedup-train.log"
GPU=1

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

if [[ ! -f "$LABELS" ]]; then
  echo "missing labels: $LABELS"
  exit 1
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo '=== ff-sft-train (1809 dedup labels, base Qwen3-4B) ===' | tee '$LOG_FILE'
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
echo "  labels: $(wc -l < $LABELS) rows from $LABELS"
echo "  gpu:    CUDA_VISIBLE_DEVICES=$GPU"
echo "  out:    $ADAPTER -> $MERGED"
