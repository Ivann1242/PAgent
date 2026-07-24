#!/usr/bin/env bash
# Fair 8k eval (baseline + ff both 8192), then resume GRPO from latest saved step.
set -euo pipefail
cd /home/ivaning/PAgent

EVAL_LOG=logs/eval-blind-ff-sft-20k-ok-128-8kfair.log
GRPO_LOG=logs/blind-ff-grpo-17k-r3-train.log
OUT_ROOT=checkpoints/eval_blind_ff_sft_20k_ok_128_8kfair
MERGED=checkpoints/blind_ff_sft_20k_ok_merged
MODEL_NAME=qwen3-4b-blind-ff-20k-ok
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}

mkdir -p logs "$OUT_ROOT"

wait_model() {
  local url=$1 name=$2
  for i in $(seq 1 90); do
    curl -sf "$url" | grep -q "$name" && { echo "  $url OK"; return 0; }
    sleep 3
  done
  echo "FAIL $url"; return 1
}

echo "=== $(date -Is) pause GRPO for fair 8k eval ===" | tee "$EVAL_LOG"
# stop GRPO train only (keep OSS if possible)
pkill -f 'python run.py ff-train --init-adapter checkpoints/blind_ff_grpo_17k_r3_adapter' 2>/dev/null || true
sleep 3

# Ensure 3 OSS on GPU0/1/2 and router on GPU3
for port in 8086; do
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
done
sleep 2

# free GPU3 completely for router
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 3 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2

# start/ensure OSS on 8006/8007/8009
if ! curl -sf http://127.0.0.1:8006/v1/models | grep -q gpt-oss-20b; then
  CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model "$OSS_MODEL" --port 8006 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 8 \
    --served-model-name gpt-oss-20b > logs/oss-eval-8006.log 2>&1 &
fi
if ! curl -sf http://127.0.0.1:8007/v1/models | grep -q gpt-oss-20b; then
  CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
    --model "$OSS_MODEL" --port 8007 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 8 \
    --served-model-name gpt-oss-20b > logs/oss-eval-8007.log 2>&1 &
fi
if ! curl -sf http://127.0.0.1:8009/v1/models | grep -q gpt-oss-20b; then
  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model "$OSS_MODEL" --port 8009 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 8 \
    --served-model-name gpt-oss-20b > logs/oss-grpo-8009.log 2>&1 &
fi

CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --port 8086 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.25 --max-model-len 8192 --max-num-seqs 8 \
  --served-model-name "$MODEL_NAME" \
  > logs/blind-ff-sft-20k-ok-serve-8kfair.log 2>&1 &

wait_model http://127.0.0.1:8086/v1/models "$MODEL_NAME" | tee -a "$EVAL_LOG"
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$EVAL_LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$EVAL_LOG"
wait_model http://127.0.0.1:8009/v1/models gpt-oss-20b | tee -a "$EVAL_LOG"

echo "=== $(date -Is) fair eval both max_tokens=8192, 1 seed, 3 OSS ===" | tee -a "$EVAL_LOG"
python eval_repeat.py \
  --repeats 1 --limit 128 \
  --router-mode ff_router \
  --router-url http://127.0.0.1:8086/v1 \
  --router-model "$MODEL_NAME" \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8009/v1 \
  --max-tokens 8192 \
  --out-root "$OUT_ROOT" \
  --eval-workers 48 \
  2>&1 | tee -a "$EVAL_LOG"

echo "=== $(date -Is) FAIR EVAL DONE ===" | tee -a "$EVAL_LOG"
cat "$OUT_ROOT/aggregate.json" | tee -a "$EVAL_LOG"

# compare quick print
python3 - <<'PY' | tee -a "$EVAL_LOG"
import json
from pathlib import Path
fair=json.load(open("checkpoints/eval_blind_ff_sft_20k_ok_128_8kfair/aggregate.json"))
old=json.load(open("checkpoints/eval_blind_ff_sft_20k_ok_128_repeat/aggregate.json"))
print("unfair(base4k/ff8k):", old["live_baseline_em_mean"], "->", old["router_em_mean"], "delta", old["delta_pp_mean"])
print("fair(both8k):      ", fair["live_baseline_em_mean"], "->", fair["router_em_mean"], "delta", fair["delta_pp_mean"])
PY

echo "=== $(date -Is) resume GRPO ===" | tee -a "$GRPO_LOG"
# free GPU3 for policy
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8086' 2>/dev/null || true
sleep 3
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 3 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2

STEP=$(python3 -c "import json; print(json.load(open('checkpoints/blind_ff_grpo_17k_r3_adapter/grpo_train_state.json'))['step'])")
NEXT=$((STEP + 1))
if [[ "$NEXT" -gt 32 ]]; then
  echo "GRPO already finished at step=$STEP" | tee -a "$GRPO_LOG"
  exit 0
fi

wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b
wait_model http://127.0.0.1:8009/v1/models gpt-oss-20b

echo "=== $(date -Is) ff-train RESUME start-step=$NEXT ===" | tee -a "$GRPO_LOG"
python run.py ff-train \
  --init-adapter checkpoints/blind_ff_grpo_17k_r3_adapter \
  --out-dir checkpoints/blind_ff_grpo_17k_r3_adapter \
  --rollout-log checkpoints/blind_ff_grpo_17k_r3_rollouts.jsonl \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8009/v1 \
  --data-file data/train.jsonl \
  --batch-size 64 --max-steps 32 --k 8 \
  --gen-batch-size 8 --gpu 3 --rollout-workers 96 \
  --reward-repeats 3 --reward-max-tokens 8192 --reward-temperature 0.0 \
  --lr 1e-6 --start-step "$NEXT" \
  2>&1 | tee -a "$GRPO_LOG"

python run.py ff-merge \
  --adapter-dir checkpoints/blind_ff_grpo_17k_r3_adapter \
  --merged-dir checkpoints/blind_ff_grpo_17k_r3_merged \
  2>&1 | tee -a "$GRPO_LOG"
echo "=== $(date -Is) ALL DONE ===" | tee -a "$GRPO_LOG"
