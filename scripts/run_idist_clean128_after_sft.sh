#!/usr/bin/env bash
# Wait for clean SFT merge, then serve router+2 solvers and eval clean-sample128.
set -euo pipefail
cd /home/ivaning/PAgent

TRAIN_LOG=${TRAIN_LOG:-logs/blind-ff-sft-qwen3-14b-nothink-clean-train.log}
MERGED=${MERGED:-checkpoints/blind_ff_sft_qwen3_14b_nothink_clean_merged}
LABELS=${LABELS:-checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_clean_train_sample128.jsonl}
OUT=${OUT:-checkpoints/eval_idist_qwen3_14b_nothink_clean_sft_train128}
LOG=${LOG:-logs/eval-idist-qwen3-14b-nothink-clean-sft-train128.log}
ROUTER_NAME=${ROUTER_NAME:-qwen3-4b-blind-ff-qwen3-14b-nothink-clean}
ROUTER_PORT=${ROUTER_PORT:-8086}
TOKENS=${TOKENS:-8192}
WORKERS=${WORKERS:-64}

mkdir -p logs "$OUT"

echo "=== $(date -Is) wait for clean SFT DONE ===" | tee "$LOG"
for i in $(seq 1 720); do  # up to ~6h
  if grep -q '=== .* DONE ===' "$TRAIN_LOG" 2>/dev/null && [[ -d "$MERGED" ]] && [[ -f "$MERGED/model.safetensors" || -f "$MERGED/model.safetensors.index.json" ]]; then
    echo "train done at iter $i" | tee -a "$LOG"
    break
  fi
  if ! tmux has-session -t pagent-blind-ff-sft-qwen3-14b-nothink-clean 2>/dev/null; then
    if grep -q '=== .* DONE ===' "$TRAIN_LOG" 2>/dev/null && [[ -d "$MERGED" ]]; then
      break
    fi
    echo "train tmux died before DONE" | tee -a "$LOG"
    tail -40 "$TRAIN_LOG" | tee -a "$LOG" || true
    exit 1
  fi
  sleep 30
done

if [[ ! -d "$MERGED" ]]; then
  echo "missing merged: $MERGED" | tee -a "$LOG"
  exit 1
fi

echo "=== $(date -Is) kill leftover vLLM ===" | tee -a "$LOG"
pids=$(ps aux | awk '/vllm.entrypoints.openai.api_server/ && !/awk/ {print $2}')
if [[ -n "${pids}" ]]; then
  kill ${pids} || true
  sleep 8
fi

echo "=== $(date -Is) serve router GPU0 :$ROUTER_PORT ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" \
  --port "$ROUTER_PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.22 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --served-model-name "$ROUTER_NAME" \
  > logs/serve-router-clean-sft-8086.log 2>&1 &

echo "=== $(date -Is) serve solvers GPU1-2 :8007-8008 (skip flaky 8009) ===" | tee -a "$LOG"
for pair in 1:8007 2:8008; do
  gpu=${pair%%:*}
  port=${pair##*:}
  CUDA_VISIBLE_DEVICES=$gpu VLLM_BATCH_INVARIANT=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model /home/ivaning/models/Qwen3-14B \
    --port "$port" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 16384 \
    --max-num-seqs 48 \
    --seed 0 \
    --served-model-name qwen3-14b \
    > "logs/serve-qwen3-14b-${port}.log" 2>&1 &
done

ok=0
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$ROUTER_PORT/v1/models" | grep -q "$ROUTER_NAME" \
     && curl -sf http://127.0.0.1:8007/v1/models | grep -q qwen3-14b \
     && curl -sf http://127.0.0.1:8008/v1/models | grep -q qwen3-14b; then
    ok=1
    break
  fi
  sleep 5
done
if [[ "$ok" -ne 1 ]]; then
  echo "servers not ready" | tee -a "$LOG"
  exit 1
fi
echo "servers ready" | tee -a "$LOG"

rm -f "$OUT"/live_baseline.jsonl "$OUT"/ff_router.jsonl "$OUT"/oracle_hint.jsonl "$OUT"/summary.json

echo "=== $(date -Is) idist clean128 n=$(wc -l < "$LABELS") ===" | tee -a "$LOG"
python eval_idist.py \
  --labels-file "$LABELS" \
  --out-dir "$OUT" \
  --router-model "$ROUTER_NAME" \
  --router-url "http://127.0.0.1:$ROUTER_PORT/v1" \
  --answer-urls http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \
  --modes live_baseline ff_router oracle_hint \
  --workers "$WORKERS" \
  --max-tokens "$TOKENS" \
  --protocol native \
  2>&1 | tee -a "$LOG"

python - <<PY | tee -a "$LOG"
import json
from pathlib import Path
s = json.loads(Path("$OUT/summary.json").read_text())
b = s["live_baseline"]["em"] * 100
r = s["ff_router"]["em"] * 100
o = s["oracle_hint"]["em"] * 100
print(f"CLEAN128 n={s['meta']['n_questions']} baseline={b:.1f}% router={r:.1f}% oracle={o:.1f}% delta={r-b:+.1f}pp")
PY
echo "=== $(date -Is) ALL DONE ===" | tee -a "$LOG"
