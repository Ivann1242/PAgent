#!/usr/bin/env bash
# Blind FF GRPO on top of 20k-ok SFT.
# batch=64, K=8, reward=mean(EM×3) at temp=0, max_tokens=8k.
# Policy on GPU0; OSS reward on GPU1/2/3 (:8007,:8008,:8006).
set -euo pipefail
cd /home/ivaning/PAgent
LOG_DIR=/home/ivaning/PAgent/logs
mkdir -p "$LOG_DIR"

SESSION=pagent-blind-ff-grpo-20k-ok
INIT_ADAPTER=${INIT_ADAPTER:-checkpoints/blind_ff_sft_20k_ok_adapter}
OUT_ADAPTER=${OUT_ADAPTER:-checkpoints/blind_ff_grpo_20k_ok_adapter}
MERGED=${MERGED:-checkpoints/blind_ff_grpo_20k_ok_merged}
ROLLOUT_LOG=${ROLLOUT_LOG:-checkpoints/blind_ff_grpo_20k_ok_rollouts.jsonl}
LOG_FILE=${LOG_FILE:-$LOG_DIR/blind-ff-grpo-20k-ok-train.log}
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
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo \"=== \$(date -Is) wait for SFT adapter: $INIT_ADAPTER ===\" | tee '$LOG_FILE'
  for i in \$(seq 1 720); do
    if [[ -f $INIT_ADAPTER/adapter_config.json && -f $INIT_ADAPTER/sft_meta.json ]]; then
      echo \"SFT adapter ready after \${i} checks\" | tee -a '$LOG_FILE'
      break
    fi
    if [[ \$i -eq 720 ]]; then
      echo 'ERROR: timed out waiting for SFT adapter' | tee -a '$LOG_FILE'
      exit 1
    fi
    sleep 30
  done

  echo \"=== \$(date -Is) start OSS reward servers (GPU1/2/3) ===\" | tee -a '$LOG_FILE'
  bash scripts/serve_oss_grpo_reward.sh 2>&1 | tee -a '$LOG_FILE'

  echo \"=== \$(date -Is) ff-train GRPO ===\" | tee -a '$LOG_FILE'
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

echo "started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG_FILE"
echo "  config: batch=$BATCH_SIZE k=$K steps=$MAX_STEPS repeats=$REWARD_REPEATS tokens=$REWARD_MAX_TOKENS temp=$REWARD_TEMP"
echo "  policy: GPU$POLICY_GPU  reward OSS: $ANSWER_URLS"
echo "  init:   $INIT_ADAPTER -> $OUT_ADAPTER"
echo "  note:   waits for SFT, then restarts OSS on GPU1/2/3"
