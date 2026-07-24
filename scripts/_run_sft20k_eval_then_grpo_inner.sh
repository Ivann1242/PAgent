#!/usr/bin/env bash
# Inner runner for pagent-sft20k-eval-then-grpo (invoked inside tmux).
set -euo pipefail
cd /home/ivaning/PAgent

EVAL_LOG=logs/eval-blind-ff-sft-20k-ok-128-repeat.log
GRPO_LOG=logs/blind-ff-grpo-17k-r3-train.log
SERVE_LOG=logs/blind-ff-sft-20k-ok-serve.log
MERGED=checkpoints/blind_ff_sft_20k_ok_merged
MODEL_NAME=qwen3-4b-blind-ff-20k-ok
OUT_ROOT=checkpoints/eval_blind_ff_sft_20k_ok_128_repeat
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}

mkdir -p logs "$OUT_ROOT"

wait_model() {
  local url=$1 name=$2
  for i in $(seq 1 120); do
    if curl -sf "$url" | grep -q "$name"; then
      echo "  $url OK"
      return 0
    fi
    sleep 5
  done
  echo "FAIL $url"
  return 1
}

echo "=== $(date -Is) PHASE1: serve router + OSS, eval 20k-ok SFT ===" | tee "$EVAL_LOG"

for port in 8086 8006 8007 8008 8009; do
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
done
sleep 5

# Router GPU0
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --port 8086 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.25 --max-model-len 8192 --max-num-seqs 8 \
  --served-model-name "$MODEL_NAME" \
  > "$SERVE_LOG" 2>&1 &
ROUTER_PID=$!
echo "router pid=$ROUTER_PID" | tee -a "$EVAL_LOG"

# OSS GPU1/2/3
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" --port 8006 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 8 \
  --served-model-name gpt-oss-20b \
  > logs/oss-eval-8006.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" --port 8007 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 8 \
  --served-model-name gpt-oss-20b \
  > logs/oss-eval-8007.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" --port 8008 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 8 \
  --served-model-name gpt-oss-20b \
  > logs/oss-eval-8008.log 2>&1 &

wait_model http://127.0.0.1:8086/v1/models "$MODEL_NAME" | tee -a "$EVAL_LOG"
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$EVAL_LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$EVAL_LOG"
wait_model http://127.0.0.1:8008/v1/models gpt-oss-20b | tee -a "$EVAL_LOG"

echo "=== $(date -Is) eval_repeat 128x4 ===" | tee -a "$EVAL_LOG"
python eval_repeat.py \
  --repeats 4 --limit 128 \
  --router-mode ff_router \
  --router-url http://127.0.0.1:8086/v1 \
  --router-model "$MODEL_NAME" \
  --out-root "$OUT_ROOT" \
  --eval-workers 32 \
  2>&1 | tee -a "$EVAL_LOG"

echo "=== $(date -Is) EVAL DONE ===" | tee -a "$EVAL_LOG"
cat "$OUT_ROOT/aggregate.json" | tee -a "$EVAL_LOG"

echo "=== $(date -Is) PHASE2: resume GRPO from step18 ===" | tee -a "$GRPO_LOG"
kill "$ROUTER_PID" 2>/dev/null || true
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8086' 2>/dev/null || true
sleep 4

# Free GPU3 for policy: stop OSS :8008
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8008' 2>/dev/null || true
sleep 3
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 3 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2

# OSS on GPU0 -> :8009
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" --port 8009 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 8 \
  --served-model-name gpt-oss-20b \
  > logs/oss-grpo-8009.log 2>&1 &

wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$GRPO_LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$GRPO_LOG"
wait_model http://127.0.0.1:8009/v1/models gpt-oss-20b | tee -a "$GRPO_LOG"

echo "=== $(date -Is) ff-train RESUME start-step=18 init=grpo@step17 ===" | tee -a "$GRPO_LOG"
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

echo "=== $(date -Is) ff-merge ===" | tee -a "$GRPO_LOG"
python run.py ff-merge \
  --adapter-dir checkpoints/blind_ff_grpo_17k_r3_adapter \
  --merged-dir checkpoints/blind_ff_grpo_17k_r3_merged \
  2>&1 | tee -a "$GRPO_LOG"

echo "=== $(date -Is) ALL DONE ===" | tee -a "$GRPO_LOG"
