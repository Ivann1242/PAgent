#!/usr/bin/env bash
# Solvers on GPU1/2 (:8007/:8008). Policy+router on GPU3, coexist with small foreign jobs (never kill).
set -euo pipefail
cd /home/ivaning/PAgent
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=logs/blind-ff-grpo-clean-iid-vp-train.log
EVAL_LOG=logs/eval-grpo-step200-clean-iid128-500.log
OUT=checkpoints/blind_ff_grpo_clean_iid_vp_adapter
MERGED=checkpoints/blind_ff_grpo_clean_iid_vp_merged
REF=checkpoints/blind_ff_sft_qwen3_14b_nothink_clean_adapter
LABELS128=checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_clean_train_sample128.jsonl
LABELS500=checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_clean_train_sample500.jsonl
ROUTER_NAME=qwen3-4b-blind-ff-grpo-clean-iid-vp-step200
ROUTER_PORT=8086
POLICY_GPU=3

mkdir -p logs

echo "=== $(date -Is) GRPO ->200 on GPU${POLICY_GPU} (coexist; never kill foreign) solvers GPU1/2 ===" | tee -a "$LOG" | tee "$EVAL_LOG"
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$POLICY_GPU" | tr -d ' ')
echo "GPU${POLICY_GPU} free_MiB=$FREE" | tee -a "$LOG" | tee -a "$EVAL_LOG"
if [[ "${FREE%.*}" -lt 40000 ]]; then
  echo "GPU${POLICY_GPU} free mem too low ($FREE); abort without killing anyone" | tee -a "$LOG"
  exit 1
fi

ok=1
for p in 8007 8008; do
  curl -sf "http://127.0.0.1:$p/v1/models" | grep -q qwen3-14b || ok=0
done
if [[ "$ok" -ne 1 ]]; then
  echo "restart OUR solvers only on GPU1/2" | tee -a "$LOG"
  pkill -f 'vllm.entrypoints.openai.api_server.*--port 8007' 2>/dev/null || true
  pkill -f 'vllm.entrypoints.openai.api_server.*--port 8008' 2>/dev/null || true
  sleep 5
  for pair in 1:8007 2:8008; do
    gpu=${pair%%:*}; port=${pair##*:}
    CUDA_VISIBLE_DEVICES=$gpu VLLM_BATCH_INVARIANT=1 nohup python -m vllm.entrypoints.openai.api_server \
      --model /home/ivaning/models/Qwen3-14B --port "$port" --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.92 --max-model-len 16384 --max-num-seqs 48 \
      --seed 0 --served-model-name qwen3-14b \
      > "logs/solver-grpo-clean-${port}.log" 2>&1 &
  done
  for p in 8007 8008; do
    for j in $(seq 1 90); do
      curl -sf "http://127.0.0.1:$p/v1/models" | grep -q qwen3-14b && break
      sleep 5
    done
  done
fi

START=$(python3 -c "import json; print(json.load(open('$OUT/grpo_train_state.json'))['step']+1)")
echo "=== $(date -Is) RESUME GRPO GPU$POLICY_GPU start=$START -> 200 ===" | tee -a "$LOG"

python run.py ff-train \
  --init-adapter "$OUT" \
  --reference-adapter "$REF" \
  --out-dir "$OUT" \
  --rollout-log checkpoints/blind_ff_grpo_clean_iid_vp_rollouts.jsonl \
  --answer-urls http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \
  --data-file data/grpo_clean_iid.jsonl \
  --batch-size 32 \
  --max-steps 200 \
  --k 8 \
  --gen-batch-size 8 \
  --gpu "$POLICY_GPU" \
  --rollout-workers 64 \
  --reward-repeats 1 \
  --reward-max-tokens 8192 \
  --reward-temperature 0.0 \
  --lr 1e-6 \
  --start-step "$START" \
  --checkpoint-every 10 \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Is) merge ===" | tee -a "$LOG" | tee -a "$EVAL_LOG"
python run.py ff-merge --adapter-dir "$OUT" --merged-dir "$MERGED" 2>&1 | tee -a "$LOG" | tee -a "$EVAL_LOG"

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$POLICY_GPU" | tr -d ' ')
echo "pre-router GPU$POLICY_GPU free_MiB=$FREE" | tee -a "$EVAL_LOG"
if [[ "${FREE%.*}" -lt 25000 ]]; then
  echo "not enough free mem for router; abort without killing" | tee -a "$EVAL_LOG"
  exit 1
fi

pkill -f "vllm.entrypoints.openai.api_server.*--port ${ROUTER_PORT}" 2>/dev/null || true
sleep 3
echo "=== $(date -Is) serve router GPU$POLICY_GPU :$ROUTER_PORT ===" | tee -a "$EVAL_LOG"
CUDA_VISIBLE_DEVICES=$POLICY_GPU VLLM_BATCH_INVARIANT=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" \
  --port "$ROUTER_PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.22 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --served-model-name "$ROUTER_NAME" \
  > logs/serve-router-grpo-step200.log 2>&1 &

for j in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$ROUTER_PORT/v1/models" | grep -q "$ROUTER_NAME" \
     && curl -sf http://127.0.0.1:8007/v1/models | grep -q qwen3-14b \
     && curl -sf http://127.0.0.1:8008/v1/models | grep -q qwen3-14b; then
    break
  fi
  sleep 5
done

for tag in 128 500; do
  if [[ "$tag" == 128 ]]; then LABELS="$LABELS128"; else LABELS="$LABELS500"; fi
  OUTD=checkpoints/eval_idist_grpo_clean_iid_vp_step200_train${tag}
  mkdir -p "$OUTD"
  rm -f "$OUTD"/live_baseline.jsonl "$OUTD"/ff_router.jsonl "$OUTD"/oracle_hint.jsonl "$OUTD"/summary.json
  echo "=== $(date -Is) eval clean${tag} ===" | tee -a "$EVAL_LOG"
  python eval_idist.py \
    --labels-file "$LABELS" \
    --out-dir "$OUTD" \
    --router-model "$ROUTER_NAME" \
    --router-url "http://127.0.0.1:$ROUTER_PORT/v1" \
    --answer-urls http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \
    --modes live_baseline ff_router oracle_hint \
    --workers 64 \
    --max-tokens 8192 \
    --protocol native \
    2>&1 | tee -a "$EVAL_LOG"
  python3 -c "import json; s=json.load(open('$OUTD/summary.json')); print(f\"CLEAN${tag} n={s['meta']['n_questions']} B={s['live_baseline']['em']*100:.1f}% R={s['ff_router']['em']*100:.1f}% O={s['oracle_hint']['em']*100:.1f}%\")" | tee -a "$EVAL_LOG"
done

echo "=== $(date -Is) ALL DONE step200+eval ===" | tee -a "$LOG" | tee -a "$EVAL_LOG"
cat "$OUT/grpo_train_state.json" | tee -a "$EVAL_LOG"
