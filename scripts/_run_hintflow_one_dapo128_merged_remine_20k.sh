#!/usr/bin/env bash
# HintFlow_one on DAPO-128 @20k with NEW merged-remine router.
# GPU0: new router :8087; GPU1: OSS :8006; GPU2: keep old HF1 :8086; skip GPU3.
set -euo pipefail
cd /home/ivaning/PAgent

LOG=logs/eval-hintflow-one-dapo128-merged-remine-20k.log
OUT=checkpoints/eval_hintflow_one_dapo128_merged_remine_20k_seed41
MERGED=checkpoints/blind_ff_sft_merged_remine_merged
ROUTER_NAME=qwen3-4b-blind-ff-merged-remine
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}
TOKENS=20000
SEED=41

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

echo "=== $(date -Is) HF1 DAPO-128 @${TOKENS} merged-remine (seed=${SEED}) ===" | tee "$LOG"

# Free only ports we need; do NOT touch :8086 (old HF1) or GPU3.
kill_port 8006
kill_port 8087
sleep 3

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --port 8087 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 32 \
  --served-model-name "$ROUTER_NAME" \
  > logs/orch-merged-remine-gpu0-8087.log 2>&1 &
echo "  new router gpu0 :8087 pid=$!" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" --port 8006 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 16 \
  --served-model-name gpt-oss-20b \
  > logs/oss-dapo128-gpu1-8006.log 2>&1 &
echo "  OSS gpu1 :8006 pid=$!" | tee -a "$LOG"

wait_model http://127.0.0.1:8087/v1/models "$ROUTER_NAME" | tee -a "$LOG"
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$LOG"
# old HF1 should still be up
if curl -sf http://127.0.0.1:8086/v1/models | grep -q qwen3-4b-blind-ff-17k; then
  echo "  old HF1 :8086 still OK" | tee -a "$LOG"
else
  echo "  WARN: old HF1 :8086 not up (continuing)" | tee -a "$LOG"
fi

python HintFlow_one/eval_one.py \
  --data-file data/DAPO-Math.parquet \
  --limit 128 \
  --out-dir "$OUT" \
  --orch-url http://127.0.0.1:8087/v1 \
  --orch-model "$ROUTER_NAME" \
  --solver-urls http://127.0.0.1:8006/v1 \
  --solver-model gpt-oss-20b \
  --solver-max-tokens "$TOKENS" \
  --selector-mode orch \
  --replace-threshold 0.90 \
  --seed "$SEED" \
  --workers 16 \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Is) DONE ===" | tee -a "$LOG"
cat "$OUT/summary.json" | tee -a "$LOG"

python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
new=json.load(open("checkpoints/eval_hintflow_one_dapo128_merged_remine_20k_seed41/summary.json"))["hintflow_one"]
old_p=Path("checkpoints/eval_hintflow_one_orch_blindff17k_gpu3_128_20k/summary.json")
print("\n=== compare vs old HF1 17k @20k seed41 ===")
def fmt(h):
    return f"base={100*h['baseline_em']:.1f}% chal={100*h['challenger_em']:.1f}% HF1={100*h['em']:.1f}% d={100*h['paired_delta']:+.1f}pp rec={h['recovered']} harm={h['harmed']}"
print("NEW merged-remine:", fmt(new))
if old_p.exists():
    old=json.load(open(old_p))["hintflow_one"]
    print("OLD 17k HF1:     ", fmt(old))
PY
