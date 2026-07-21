#!/usr/bin/env bash
# Blind FF GRPO from existing 17k SFT adapter.
# Layout: GPU0=policy, GPU1/2/3=OSS reward (:8007,:8008,:8006)
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=pagent-blind-ff-grpo-17k
INIT_ADAPTER=${INIT_ADAPTER:-checkpoints/blind_ff_sft_17k_adapter}
OUT_ADAPTER=${OUT_ADAPTER:-checkpoints/blind_ff_grpo_17k_r3_adapter}
MERGED=${MERGED:-checkpoints/blind_ff_grpo_17k_r3_merged}
ROLLOUT_LOG=${ROLLOUT_LOG:-checkpoints/blind_ff_grpo_17k_r3_rollouts.jsonl}
LOG_FILE=${LOG_FILE:-$LOG_DIR/blind-ff-grpo-17k-r3-train.log}
POLICY_GPU=${POLICY_GPU:-0}
MAX_STEPS=${MAX_STEPS:-32}
BATCH_SIZE=${BATCH_SIZE:-64}
K=${K:-8}
REWARD_REPEATS=${REWARD_REPEATS:-3}
REWARD_MAX_TOKENS=${REWARD_MAX_TOKENS:-8192}
REWARD_TEMP=${REWARD_TEMP:-0.0}
GEN_BATCH=${GEN_BATCH:-8}
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-96}
ANSWER_URLS=${ANSWER_URLS:-http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8006/v1}
DATA_FILE=${DATA_FILE:-data/train.jsonl}
LR=${LR:-1e-6}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo \"=== \$(date -Is) precheck OSS reward endpoints ===\" | tee '$LOG_FILE'
  for url in http://127.0.0.1:8007/v1 http://127.0.0.1:8008/v1 http://127.0.0.1:8006/v1; do
    curl -sf \"\${url}/models\" | grep -q gpt-oss-20b
    echo \"  \$url OK\" | tee -a '$LOG_FILE'
  done

  echo \"=== \$(date -Is) ff-train GRPO from $INIT_ADAPTER ===\" | tee -a '$LOG_FILE'
  python run.py ff-train \\
    --init-adapter $INIT_ADAPTER \\
    --out-dir $OUT_ADAPTER \\
    --rollout-log $ROLLOUT_LOG \\
    --answer-urls '$ANSWER_URLS' \\
    --data-file $DATA_FILE \\
    --batch-size $BATCH_SIZE \\
    --max-steps $MAX_STEPS \\
    --k $K \\
    --gen-batch-size $GEN_BATCH \\
    --gpu $POLICY_GPU \\
    --rollout-workers $ROLLOUT_WORKERS \\
    --reward-repeats $REWARD_REPEATS \\
    --reward-max-tokens $REWARD_MAX_TOKENS \\
    --reward-temperature $REWARD_TEMP \\
    --lr $LR \\
    2>&1 | tee -a '$LOG_FILE'

  echo \"=== \$(date -Is) ff-merge ===\" | tee -a '$LOG_FILE'
  python run.py ff-merge \\
    --adapter-dir $OUT_ADAPTER \\
    --merged-dir $MERGED \\
    2>&1 | tee -a '$LOG_FILE'

  echo \"=== \$(date -Is) DONE ===\" | tee -a '$LOG_FILE'
  cat $OUT_ADAPTER/grpo_train_state.json | tee -a '$LOG_FILE'
"

echo "started tmux: $SESSION"
echo "  log: tail -f $LOG_FILE"
echo "  layout: GPU0=policy | GPU1=:8007 GPU2=:8008 GPU3=:8006 OSS"
echo "  init: $INIT_ADAPTER  steps=$MAX_STEPS batch=$BATCH_SIZE k=$K repeats=$REWARD_REPEATS"
