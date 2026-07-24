#!/usr/bin/env bash
# Fair 4k eval of blind_ff_sft_17k_merged (both baseline & router @4096), then resume GRPO.
set -euo pipefail
cd /home/ivaning/PAgent

EVAL_LOG=logs/eval-blind-ff-sft-17k-128-4kfair.log
GRPO_LOG=logs/blind-ff-grpo-17k-r3-train.log
OUT_ROOT=checkpoints/eval_blind_ff_sft_17k_128_4kfair
MERGED=checkpoints/blind_ff_sft_17k_merged
MODEL_NAME=qwen3-4b-blind-ff-17k

mkdir -p logs "$OUT_ROOT"

wait_model() {
  local url=$1 name=$2
  for i in $(seq 1 90); do
    curl -sf "$url" | grep -q "$name" && { echo "  $url OK"; return 0; }
    sleep 3
  done
  echo "FAIL $url"; return 1
}

echo "=== $(date -Is) pause GRPO; fair 4k eval of 17k SFT ===" | tee "$EVAL_LOG"
pkill -f 'python run.py ff-train --init-adapter checkpoints/blind_ff_grpo_17k_r3_adapter' 2>/dev/null || true
sleep 3

pkill -f 'vllm.entrypoints.openai.api_server.*--port 8086' 2>/dev/null || true
sleep 2
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 3 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2

CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --port 8086 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.25 --max-model-len 8192 --max-num-seqs 8 \
  --served-model-name "$MODEL_NAME" \
  > logs/blind-ff-sft-17k-serve-4kfair.log 2>&1 &

wait_model http://127.0.0.1:8086/v1/models "$MODEL_NAME" | tee -a "$EVAL_LOG"
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$EVAL_LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$EVAL_LOG"
wait_model http://127.0.0.1:8009/v1/models gpt-oss-20b | tee -a "$EVAL_LOG"

echo "=== $(date -Is) eval_repeat both max_tokens=4096 ===" | tee -a "$EVAL_LOG"
python eval_repeat.py \
  --repeats 1 --limit 128 \
  --router-mode ff_router \
  --router-url http://127.0.0.1:8086/v1 \
  --router-model "$MODEL_NAME" \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8009/v1 \
  --max-tokens 4096 \
  --out-root "$OUT_ROOT" \
  --eval-workers 48 \
  2>&1 | tee -a "$EVAL_LOG"

echo "=== $(date -Is) EVAL DONE ===" | tee -a "$EVAL_LOG"
cat "$OUT_ROOT/aggregate.json" | tee -a "$EVAL_LOG"

python3 - <<'PY' | tee -a "$EVAL_LOG"
import json
from pathlib import Path
def get(p):
    p=Path(p)
    return json.load(open(p)) if p.exists() else None
for name,p in [
    ("SFT17k fair4k", "checkpoints/eval_blind_ff_sft_17k_128_4kfair/aggregate.json"),
    ("SFT17k fair8k", "checkpoints/eval_blind_ff_sft_17k_128_8kfair/aggregate.json"),
    ("SFT17k old unfair (base4k/ff8k)", "checkpoints/eval_blind_ff_sft_17k_128_repeat/aggregate.json"),
]:
    a=get(p)
    if not a: print(name, "MISSING"); continue
    if a.get("repeats")==1 and a.get("runs"):
        r=a["runs"][0]
        print(f"{name}: base={r['live_baseline_em']:.4f} ff={r['router_em']:.4f} d={r['delta_pp']:+.2f}pp")
    else:
        print(f"{name}: base={a['live_baseline_em_mean']:.4f} ff={a['router_em_mean']:.4f} d={a['delta_pp_mean']:+.2f}pp")
PY

echo "=== $(date -Is) resume GRPO ===" | tee -a "$GRPO_LOG"
pkill -f 'vllm.entrypoints.openai.api_server.*--port 8086' 2>/dev/null || true
sleep 3
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 3 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2

STEP=$(python3 -c "import json; print(json.load(open('checkpoints/blind_ff_grpo_17k_r3_adapter/grpo_train_state.json'))['step'])")
NEXT=$((STEP + 1))
echo "resume start-step=$NEXT (saved step=$STEP)" | tee -a "$GRPO_LOG"
if [[ "$NEXT" -gt 32 ]]; then
  echo "already done"; exit 0
fi

wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b
wait_model http://127.0.0.1:8009/v1/models gpt-oss-20b

python run.py ff-train \
  --init-adapter checkpoints/blind_ff_grpo_17k_r3_adapter \
  --out-dir checkpoints/blind_ff_grpo_17k_r3_adapter \
  --rollout-log checkpoints/blind_ff_grpo_17k_r3_rollouts.jsonl \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8009/v1 \
  --data-file data/train.jsonl \
  --batch-size 64 --max-steps 32 --k 8 --gen-batch-size 8 \
  --gpu 3 --rollout-workers 96 \
  --reward-repeats 3 --reward-max-tokens 8192 --reward-temperature 0.0 \
  --lr 1e-6 --start-step "$NEXT" \
  2>&1 | tee -a "$GRPO_LOG"
