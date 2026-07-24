#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent
EVAL_LOG=logs/eval-blind-ff-sft-20k-ok-128-repeat.log
GRPO_LOG=logs/blind-ff-grpo-17k-r3-train.log
OUT_ROOT=checkpoints/eval_blind_ff_sft_20k_ok_128_repeat

wait_model() {
  local url=$1 name=$2
  for i in $(seq 1 60); do
    curl -sf "$url" | grep -q "$name" && return 0
    sleep 3
  done
  return 1
}

echo "=== $(date -Is) FAST eval: 1 seed, 3 OSS, workers=48 ===" | tee "$EVAL_LOG"
python eval_repeat.py \
  --repeats 1 --limit 128 \
  --router-mode ff_router \
  --router-url http://127.0.0.1:8086/v1 \
  --router-model qwen3-4b-blind-ff-20k-ok \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \
  --out-root "$OUT_ROOT" \
  --eval-workers 48 \
  2>&1 | tee -a "$EVAL_LOG"

echo "=== $(date -Is) EVAL DONE ===" | tee -a "$EVAL_LOG"
cat "$OUT_ROOT/aggregate.json" | tee -a "$EVAL_LOG"

echo "=== $(date -Is) PHASE2: resume GRPO step18 ===" | tee -a "$GRPO_LOG"
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8086' 2>/dev/null || true
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8008' 2>/dev/null || true
sleep 4
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 3 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2

if ! curl -sf http://127.0.0.1:8009/v1/models | grep -q gpt-oss-20b; then
  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model /home/ivaning/models/gpt-oss-20b --port 8009 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 8 \
    --served-model-name gpt-oss-20b \
    > logs/oss-grpo-8009.log 2>&1 &
fi
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b
wait_model http://127.0.0.1:8009/v1/models gpt-oss-20b

echo "=== $(date -Is) ff-train RESUME start-step=18 ===" | tee -a "$GRPO_LOG"
python run.py ff-train \
  --init-adapter checkpoints/blind_ff_grpo_17k_r3_adapter \
  --out-dir checkpoints/blind_ff_grpo_17k_r3_adapter \
  --rollout-log checkpoints/blind_ff_grpo_17k_r3_rollouts.jsonl \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8009/v1 \
  --data-file data/train.jsonl \
  --batch-size 64 --max-steps 32 --k 8 \
  --gen-batch-size 8 --gpu 3 --rollout-workers 96 \
  --reward-repeats 3 --reward-max-tokens 8192 --reward-temperature 0.0 \
  --lr 1e-6 --start-step 18 \
  2>&1 | tee -a "$GRPO_LOG"

python run.py ff-merge \
  --adapter-dir checkpoints/blind_ff_grpo_17k_r3_adapter \
  --merged-dir checkpoints/blind_ff_grpo_17k_r3_merged \
  2>&1 | tee -a "$GRPO_LOG"
echo "=== $(date -Is) ALL DONE ===" | tee -a "$GRPO_LOG"
