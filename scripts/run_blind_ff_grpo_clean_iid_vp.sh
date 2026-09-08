#!/usr/bin/env bash
# Virtual-positive GRPO on clean IID (true baseline=0 + verified flip labels).
# Init from clean SFT adapter. Reward solvers: Qwen3-14B on GPU1/2 (skip flaky GPU3).
# Pilot default: 40 steps.
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=${SESSION:-pagent-blind-ff-grpo-clean-iid}
LOG=${LOG:-logs/blind-ff-grpo-clean-iid-vp-train.log}
SFT_INIT=${SFT_INIT:-checkpoints/blind_ff_sft_qwen3_14b_nothink_clean_adapter}
REF=${REF:-checkpoints/blind_ff_sft_qwen3_14b_nothink_clean_adapter}
OUT=${OUT:-checkpoints/blind_ff_grpo_clean_iid_vp_adapter}
MERGED=${MERGED:-checkpoints/blind_ff_grpo_clean_iid_vp_merged}
ROLLOUT=${ROLLOUT:-checkpoints/blind_ff_grpo_clean_iid_vp_rollouts.jsonl}
DATA=${DATA:-data/grpo_clean_iid.jsonl}
MAX_STEPS=${MAX_STEPS:-40}
BATCH=${BATCH:-32}
K=${K:-8}
GPU_POLICY=${GPU_POLICY:-0}

mkdir -p logs "$OUT"

if [[ ! -f "$DATA" ]]; then
  echo "missing data: $DATA" >&2
  exit 1
fi
if [[ ! -d "$SFT_INIT" ]]; then
  echo "missing init adapter: $SFT_INIT" >&2
  exit 1
fi

STATE="$OUT/grpo_train_state.json"
if [[ -f "$STATE" ]]; then
  SAVED_STEP=$(python -c "import json; print(json.load(open('$STATE'))['step'])")
  START_STEP=$((SAVED_STEP + 1))
  INIT="$OUT"
else
  SAVED_STEP=0
  START_STEP=1
  INIT="$SFT_INIT"
fi

wait_model() {
  local url=$1 name=$2
  for _ in $(seq 1 120); do
    if curl -sf "$url/models" | python -c \
      "import json,sys; assert any(x['id']=='$name' for x in json.load(sys.stdin)['data'])" \
      >/dev/null 2>&1; then
      echo "  $url $name OK"
      return 0
    fi
    sleep 3
  done
  echo "FAIL $url $name"
  return 1
}

echo "=== $(date -Is) prepare reward solvers (GPU1/2, skip 8009) ==="
# free router on 8086 + any old solvers; leave GPU0 for policy
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8086' 2>/dev/null || true
for port in 8006 8007 8008 8009; do
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}" || true)
  if [[ -n "${pids}" ]]; then kill ${pids} 2>/dev/null || true; fi
done
sleep 6

MODEL="${SOLVER_MODEL_PATH:-/home/ivaning/models/Qwen3-14B}"
SERVED=qwen3-14b
for pair in 1:8007 2:8008; do
  gpu=${pair%%:*}; port=${pair##*:}
  CUDA_VISIBLE_DEVICES=$gpu VLLM_BATCH_INVARIANT=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$port" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 --max-model-len 16384 --max-num-seqs 48 \
    --seed 0 --served-model-name "$SERVED" \
    > "logs/solver-grpo-clean-${port}.log" 2>&1 &
  echo "  started GPU${gpu}->:${port}"
done

wait_model http://127.0.0.1:8007/v1 "$SERVED"
wait_model http://127.0.0.1:8008/v1 "$SERVED"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd /home/ivaning/PAgent
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  : > '$LOG'

  echo \"=== \$(date -Is) GRPO-VP clean-IID pilot ===\" | tee -a '$LOG'
  echo \"data=$DATA n=\$(wc -l < $DATA)\" | tee -a '$LOG'
  echo \"init=$INIT ref=$REF out=$OUT\" | tee -a '$LOG'
  echo \"batch=$BATCH K=$K steps=$MAX_STEPS start=$START_STEP gpu_policy=$GPU_POLICY\" | tee -a '$LOG'
  echo 'reward: 8007+8008 qwen3-14b; skip flaky 8009' | tee -a '$LOG'

  python run.py ff-train \
    --init-adapter '$INIT' \
    --reference-adapter '$REF' \
    --out-dir '$OUT' \
    --rollout-log '$ROLLOUT' \
    --answer-urls 'http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1' \
    --data-file '$DATA' \
    --batch-size $BATCH \
    --max-steps $MAX_STEPS \
    --k $K \
    --gen-batch-size 8 \
    --gpu $GPU_POLICY \
    --rollout-workers 64 \
    --reward-repeats 1 \
    --reward-max-tokens 8192 \
    --reward-temperature 0.0 \
    --lr 1e-6 \
    --start-step '$START_STEP' \
    --checkpoint-every 10 \
    2>&1 | tee -a '$LOG'

  echo \"=== \$(date -Is) merge ===\" | tee -a '$LOG'
  python run.py ff-merge \
    --adapter-dir '$OUT' \
    --merged-dir '$MERGED' \
    2>&1 | tee -a '$LOG'

  echo \"=== \$(date -Is) DONE ===\" | tee -a '$LOG'
  cat '$OUT/grpo_train_state.json' | tee -a '$LOG' || true
"

echo "started tmux: $SESSION"
echo "  log: tail -f $LOG"
echo "  data: $DATA ($(wc -l < "$DATA") qs)"
echo "  out:  $OUT"
