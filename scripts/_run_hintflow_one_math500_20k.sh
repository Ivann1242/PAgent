#!/usr/bin/env bash
# HintFlow_one OOD on HuggingFaceH4/MATH-500 (500 problems), fair 20k.
# Layout: GPU0-1 OSS solvers, GPU2 Blind FF router. GPU3 left alone (other user).
set -euo pipefail
cd /home/ivaning/PAgent

LOG=logs/eval-hintflow-one-math500-20k.log
OUT=checkpoints/eval_hintflow_one_math500_20k
DATA=data/math500.jsonl
ROUTER_NAME=qwen3-4b-blind-ff-17k

mkdir -p logs "$OUT"

wait_model() {
  local url=$1 name=$2
  for i in $(seq 1 60); do
    curl -sf "$url" | grep -q "$name" && { echo "  $url OK"; return 0; }
    sleep 2
  done
  echo "FAIL $url"; return 1
}

echo "=== $(date -Is) HintFlow_one MATH-500 @20k (2 solvers + router; GPU3 skipped) ===" | tee "$LOG"
echo "GPU layout:" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee -a "$LOG"
nvidia-smi -i 3 --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$LOG"

wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8086/v1/models "$ROUTER_NAME" | tee -a "$LOG"

# 2 solvers × max-num-seqs=16 → keep workers near 32 to saturate both GPUs
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
  --workers 32 \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Is) DONE ===" | tee -a "$LOG"
cat "$OUT/summary.json" | tee -a "$LOG"
