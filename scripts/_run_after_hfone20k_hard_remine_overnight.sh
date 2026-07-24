#!/usr/bin/env bash
# After HintFlow_one iid@20k finishes: free GPU3, start 4x OSS, remine hard-noflip pool overnight.
# Pool = old baseline-wrong & no flip label (3397 ids). Re-baseline @8k + k=12 blind flips.
set -euo pipefail
cd /home/ivaning/PAgent

HF_OUT=checkpoints/eval_hintflow_one_idist_1809_20k
LOG=logs/hard-remine-overnight.log
OUT=checkpoints/blind_hint_hard_remine_8k_k12
IDS=checkpoints/blind_hint_17k/hard_noflip_ids.txt
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}
TOKENS=8192
K=12
WORKERS=64

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

echo "=== $(date -Is) wait for HF1 iid@20k summary ===" | tee "$LOG"
while [[ ! -f "$HF_OUT/summary.json" ]]; do
  sleep 60
  if ! pgrep -f 'HintFlow_one/eval_one.py' >/dev/null 2>&1; then
    if [[ -f "$HF_OUT/summary.json" ]]; then
      break
    fi
    echo "WARN $(date -Is): eval_one not running and no summary yet" | tee -a "$LOG"
  fi
done
echo "HF1 ready:" | tee -a "$LOG"
python3 -c "import json; print(json.dumps(json.load(open('$HF_OUT/summary.json'))['hintflow_one'], indent=2))" | tee -a "$LOG"

echo "=== $(date -Is) restart 4x OSS @32k for labeling ===" | tee -a "$LOG"
for port in 8006 8007 8008 8009 8086; do kill_port "$port"; done
sleep 4
for gpu in 0 1 2 3; do
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
  done
done
sleep 4

serve_oss() {
  local gpu=$1 port=$2
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$OSS_MODEL" --port "$port" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 16 \
    --served-model-name gpt-oss-20b \
    > "logs/oss-hardremine-gpu${gpu}-${port}.log" 2>&1 &
  echo "  OSS gpu$gpu :$port pid=$!" | tee -a "$LOG"
}
serve_oss 0 8006
serve_oss 1 8007
serve_oss 2 8008
serve_oss 3 8009
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8008/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8009/v1/models gpt-oss-20b | tee -a "$LOG"

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
echo "hard ids: $(wc -l < "$IDS")" | tee -a "$LOG"

echo "=== $(date -Is) hard remine @${TOKENS} k=${K} ===" | tee -a "$LOG"
python run.py oracle-hint \
  --data-file data/train.jsonl \
  --only-ids-file "$IDS" \
  --out-dir "$OUT" \
  --workers "$WORKERS" \
  --k "$K" \
  --hint-temp 0.9 \
  --max-tokens "$TOKENS" \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1 \
  --protocol native \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Is) DONE ===" | tee -a "$LOG"
cat "$OUT/stats.json" | tee -a "$LOG"
