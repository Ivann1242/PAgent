#!/usr/bin/env bash
# Resume HintFlow_one iid@20k from partial jsonl, then run hard-noflip remine overnight.
set -euo pipefail
cd /home/ivaning/PAgent

HF_LOG=logs/eval-hintflow-one-idist-20k.log
HF_OUT=checkpoints/eval_hintflow_one_idist_1809_20k
LABELS=checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl
MERGED=checkpoints/blind_ff_sft_17k_merged
ROUTER_NAME=qwen3-4b-blind-ff-17k
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}
REMINE_LOG=logs/hard-remine-overnight.log
REMINE_OUT=checkpoints/blind_hint_hard_remine_8k_k12
IDS=checkpoints/blind_hint_17k/hard_noflip_ids.txt

mkdir -p logs "$HF_OUT" "$REMINE_OUT"

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

echo "=== $(date -Is) resume HF1 iid@20k (partial=$(wc -l < "$HF_OUT/hintflow_one.jsonl" 2>/dev/null || echo 0)) ===" | tee -a "$HF_LOG"

for port in 8006 8007 8008 8009 8086; do kill_port "$port"; done
sleep 3
for gpu in 0 1 2 3; do
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
  done
done
sleep 3

serve_oss() {
  local gpu=$1 port=$2
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$OSS_MODEL" --port "$port" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 16 \
    --served-model-name gpt-oss-20b \
    > "logs/oss-hfone20k-resume-gpu${gpu}-${port}.log" 2>&1 &
  echo "  OSS gpu$gpu :$port pid=$!"
}
serve_oss 0 8006
serve_oss 1 8007
serve_oss 2 8008
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$HF_LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$HF_LOG"
wait_model http://127.0.0.1:8008/v1/models gpt-oss-20b | tee -a "$HF_LOG"

CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --port 8086 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 16 \
  --served-model-name "$ROUTER_NAME" \
  > logs/orch-hfone20k-resume-gpu3-8086.log 2>&1 &
echo "  router gpu3 :8086 pid=$!" | tee -a "$HF_LOG"
wait_model http://127.0.0.1:8086/v1/models "$ROUTER_NAME" | tee -a "$HF_LOG"

echo "=== $(date -Is) eval_one resume ===" | tee -a "$HF_LOG"
python HintFlow_one/eval_one.py \
  --data-file "$LABELS" \
  --out-dir "$HF_OUT" \
  --orch-url http://127.0.0.1:8086/v1 \
  --orch-model "$ROUTER_NAME" \
  --solver-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \
  --solver-model gpt-oss-20b \
  --solver-max-tokens 20000 \
  --selector-mode orch \
  --replace-threshold 0.90 \
  --seed 41 \
  --workers 48 \
  2>&1 | tee -a "$HF_LOG"

echo "=== $(date -Is) HF1 DONE ===" | tee -a "$HF_LOG"
cat "$HF_OUT/summary.json" | tee -a "$HF_LOG"

echo "=== $(date -Is) start hard remine (4x OSS) ===" | tee "$REMINE_LOG"
kill_port 8086
sleep 3
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 3 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2

CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" --port 8009 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 16 \
  --served-model-name gpt-oss-20b \
  > logs/oss-hardremine-gpu3-8009.log 2>&1 &
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$REMINE_LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$REMINE_LOG"
wait_model http://127.0.0.1:8008/v1/models gpt-oss-20b | tee -a "$REMINE_LOG"
wait_model http://127.0.0.1:8009/v1/models gpt-oss-20b | tee -a "$REMINE_LOG"

if [[ ! -f "$IDS" ]]; then
  python3 - <<'PY'
import json
from pathlib import Path
base={r["id"]:r for r in map(json.loads, open("checkpoints/blind_hint_17k/baselines.jsonl"))}
lab={json.loads(l)["id"] for l in open("checkpoints/blind_hint_17k/oracle_labels.jsonl")}
hard=sorted(i for i,r in base.items() if r.get("em")==0 and i not in lab)
Path("checkpoints/blind_hint_17k/hard_noflip_ids.txt").write_text("\n".join(map(str,hard))+"\n")
print("hard_noflip", len(hard))
PY
fi

python run.py oracle-hint \
  --data-file data/train.jsonl \
  --only-ids-file "$IDS" \
  --out-dir "$REMINE_OUT" \
  --workers 64 \
  --k 12 \
  --hint-temp 0.9 \
  --max-tokens 8192 \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1 \
  --protocol native \
  2>&1 | tee -a "$REMINE_LOG"

echo "=== $(date -Is) ALL DONE ===" | tee -a "$REMINE_LOG"
cat "$REMINE_OUT/stats.json" | tee -a "$REMINE_LOG"
