#!/usr/bin/env bash
# IID 1809 @20k: live_baseline (4x OSS) then ff_router (3x OSS + Blind FF router).
# Stops GRPO; uses all 4 GPUs. Does NOT auto-resume GRPO.
set -euo pipefail
cd /home/ivaning/PAgent

LOG=logs/eval-idist-baseline-router-20k-4gpu.log
LABELS=checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl
OUT=checkpoints/eval_idist_blind_ff_17k_full_20k
MERGED=checkpoints/blind_ff_sft_17k_merged
ROUTER_NAME=qwen3-4b-blind-ff-17k
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}
TOKENS=20000
OSS_MAX_LEN=32768
# Keep concurrency near GPU capacity; huge worker>>seqs queues caused 600s client timeouts.
MAX_NUM_SEQS=16
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

echo "=== $(date -Is) stop GRPO + old serves ===" | tee "$LOG"
pkill -f 'python run.py ff-train --init-adapter checkpoints/blind_ff_grpo_17k_r3_adapter' 2>/dev/null || true
sleep 3
for port in 8006 8007 8008 8009 8086; do kill_port "$port"; done
sleep 3
# clear leftover GPU compute apps
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
    --gpu-memory-utilization 0.92 --max-model-len "$OSS_MAX_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --served-model-name gpt-oss-20b \
    > "logs/oss-idist20k-gpu${gpu}-${port}.log" 2>&1 &
  echo "  OSS gpu$gpu :$port pid=$!"
}

echo "=== $(date -Is) phase1: 4x OSS @${OSS_MAX_LEN} for baseline ===" | tee -a "$LOG"
serve_oss 0 8006
serve_oss 1 8007
serve_oss 2 8008
serve_oss 3 8009
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8008/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8009/v1/models gpt-oss-20b | tee -a "$LOG"

echo "=== $(date -Is) iid live_baseline @${TOKENS} (4 OSS) ===" | tee -a "$LOG"
python eval_idist.py \
  --labels-file "$LABELS" \
  --out-dir "$OUT" \
  --router-model "$ROUTER_NAME" \
  --router-url http://127.0.0.1:8086/v1 \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1 \
  --max-tokens "$TOKENS" \
  --workers "$WORKERS" \
  --modes live_baseline \
  2>&1 | tee -a "$LOG"

python3 - <<PY | tee -a "$LOG"
import json
s=json.load(open("$OUT/summary.json"))
print("baseline@", s["meta"].get("max_tokens"),
      f"{s['live_baseline']['em']*100:.2f}%",
      "n=", s["meta"]["n_questions"])
PY

echo "=== $(date -Is) phase2: free GPU3 for router; keep 3 OSS ===" | tee -a "$LOG"
kill_port 8009
sleep 3
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 3 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 3

CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --port 8086 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 16 \
  --served-model-name "$ROUTER_NAME" \
  > logs/orch-idist20k-gpu3-8086.log 2>&1 &
echo "  router gpu3 :8086 pid=$!" | tee -a "$LOG"
wait_model http://127.0.0.1:8086/v1/models "$ROUTER_NAME" | tee -a "$LOG"
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8008/v1/models gpt-oss-20b | tee -a "$LOG"

echo "=== $(date -Is) iid ff_router @${TOKENS} (3 OSS + router) ===" | tee -a "$LOG"
# Merge into same OUT: re-run with both modes would redo baseline; only ff_router here,
# then stitch summary.
python eval_idist.py \
  --labels-file "$LABELS" \
  --out-dir "${OUT}_router_tmp" \
  --router-model "$ROUTER_NAME" \
  --router-url http://127.0.0.1:8086/v1 \
  --answer-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \
  --max-tokens "$TOKENS" \
  --workers "$WORKERS" \
  --modes ff_router \
  2>&1 | tee -a "$LOG"

python3 - <<'PY' | tee -a "$LOG"
import json, shutil
from pathlib import Path
out = Path("checkpoints/eval_idist_blind_ff_17k_full_20k")
tmp = Path("checkpoints/eval_idist_blind_ff_17k_full_20k_router_tmp")
base = json.load(open(out / "summary.json"))
rout = json.load(open(tmp / "summary.json"))
shutil.copy(tmp / "ff_router.jsonl", out / "ff_router.jsonl")
base["ff_router"] = rout["ff_router"]
base["meta"]["modes"] = ["live_baseline", "ff_router"]
base["meta"]["answer_urls_baseline"] = base["meta"].get("answer_urls")
base["meta"]["answer_urls_router"] = rout["meta"].get("answer_urls")
base["meta"]["router_url"] = rout["meta"].get("router_url")
base["meta"]["router_model"] = rout["meta"].get("router_model")
base["ff_router_vs_live_baseline"] = (
    base["ff_router"]["em"] - base["live_baseline"]["em"]
)
# attach prior oracle@20k if present
oracle_p = Path("checkpoints/eval_idist_oraclehint_20k/summary.json")
if oracle_p.exists():
    o = json.load(open(oracle_p))
    base["oracle_hint_20k_ref"] = o["oracle_hint"]
    base["ff_router_vs_oracle_hint_20k"] = (
        base["ff_router"]["em"] - o["oracle_hint"]["em"]
    )
(out / "summary.json").write_text(json.dumps(base, indent=2) + "\n")
print(json.dumps({
    "n": base["meta"]["n_questions"],
    "max_tokens": base["meta"]["max_tokens"],
    "baseline": f"{base['live_baseline']['em']*100:.2f}%",
    "router": f"{base['ff_router']['em']*100:.2f}%",
    "delta": f"{base['ff_router_vs_live_baseline']*100:+.2f}pp",
    "oracle20k_ref": (
        f"{base['oracle_hint_20k_ref']['em']*100:.2f}%"
        if "oracle_hint_20k_ref" in base else "n/a"
    ),
}, indent=2))
shutil.rmtree(tmp, ignore_errors=True)
PY

echo "=== $(date -Is) ALL DONE ===" | tee -a "$LOG"
echo "summary: $OUT/summary.json" | tee -a "$LOG"
