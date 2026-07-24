#!/usr/bin/env bash
# HintFlow_one OOD on math-ai/aime24 (30 problems), fair 20k.
# Uses GPU0-1 for OSS solvers; GPU2 for Blind FF router. Does NOT touch GPU3.
set -euo pipefail
cd /home/ivaning/PAgent

LOG=logs/eval-hintflow-one-aime24-20k.log
OUT=checkpoints/eval_hintflow_one_aime24_20k
DATA=data/aime24.jsonl
MERGED=checkpoints/blind_ff_sft_17k_merged
ROUTER_NAME=qwen3-4b-blind-ff-17k
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}

mkdir -p logs "$OUT"

wait_model() {
  local url=$1 name=$2
  for i in $(seq 1 120); do
    curl -sf "$url" | grep -q "$name" && { echo "  $url OK"; return 0; }
    sleep 3
  done
  echo "FAIL $url"; return 1
}

kill_port() {
  pkill -f "vllm.entrypoints.openai.api_server.*--port $1" 2>/dev/null || true
}

echo "=== $(date -Is) HintFlow_one AIME24 @20k (GPU0-1 OSS, GPU2 router; skip GPU3) ===" | tee "$LOG"

# Free GPU2 for router; keep 8006/8007. Never kill GPU3 / :8009.
kill_port 8008
kill_port 8086
sleep 3

# If :8006/:8007 already healthy, keep them; else (re)start on GPU0/1.
if ! curl -sf http://127.0.0.1:8006/v1/models | grep -q gpt-oss-20b; then
  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model "$OSS_MODEL" --port 8006 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 16 \
    --served-model-name gpt-oss-20b \
    > logs/oss-aime24-gpu0-8006.log 2>&1 &
fi
if ! curl -sf http://127.0.0.1:8007/v1/models | grep -q gpt-oss-20b; then
  CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model "$OSS_MODEL" --port 8007 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 16 \
    --served-model-name gpt-oss-20b \
    > logs/oss-aime24-gpu1-8007.log 2>&1 &
fi

CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --port 8086 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 32 \
  --served-model-name "$ROUTER_NAME" \
  > logs/orch-aime24-gpu2-8086.log 2>&1 &
echo "  router gpu2 :8086 pid=$!" | tee -a "$LOG"

wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8086/v1/models "$ROUTER_NAME" | tee -a "$LOG"

# Confirm GPU3 untouched
echo "GPU3 apps (should be others only):" | tee -a "$LOG"
nvidia-smi -i 3 --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$LOG"

python HintFlow_one/eval_one.py \
  --data-file "$DATA" \
  --out-dir "$OUT" \
  --orch-url http://127.0.0.1:8086/v1 \
  --orch-model "$ROUTER_NAME" \
  --solver-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1 \
  --solver-model gpt-oss-20b \
  --solver-max-tokens 20000 \
  --selector-mode orch \
  --replace-threshold 0.90 \
  --seed 41 \
  --workers 16 \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Is) DONE ===" | tee -a "$LOG"
cat "$OUT/summary.json" | tee -a "$LOG"
