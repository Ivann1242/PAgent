#!/usr/bin/env bash
# Hint-gated single-OSS: bare then gated on IID 1809 @8k (4 solvers + orch).
# Reuses existing :8086/:8006-8009 if healthy; otherwise starts them.
set -euo pipefail
cd /home/ivaning/PAgent

MASTER_LOG=${MASTER_LOG:-logs/hintflow-gate-idist8k-4gpu.log}
LABELS=checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl
OUT=checkpoints/eval_hintflow_gate_idist_1809_8k
MERGED=${MERGED:-checkpoints/blind_ff_sft_17k_merged}
ROUTER_NAME=${ROUTER_NAME:-qwen3-4b-blind-ff-17k}
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}
TOKENS=8192
OSS_MAX_LEN=16384
ORCH_MAX_LEN=8192
WORKERS=${WORKERS:-80}
SEED=41
THRESH=${THRESH:-0.90}
SOLVER_URLS=http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1

mkdir -p logs "$OUT"

wait_model() {
  local url=$1 name=$2
  for _ in $(seq 1 180); do
    if curl -sf "$url/models" | grep -q "$name"; then
      echo "  $url $name OK"
      return 0
    fi
    sleep 3
  done
  echo "FAIL $url $name" >&2
  return 1
}

kill_port() {
  pkill -f "vllm.entrypoints.openai.api_server.*--port $1" 2>/dev/null || true
}

all_up() {
  for p in 8086 8006 8007 8008 8009; do
    curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null || return 1
  done
  return 0
}

{
  echo "=== $(date -Is) HintFlow_gate IID@8k start ==="
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
} | tee "$MASTER_LOG"

if all_up; then
  echo "=== $(date -Is) reusing existing orch+4xOSS ===" | tee -a "$MASTER_LOG"
else
  echo "=== $(date -Is) (re)starting orch+4xOSS ===" | tee -a "$MASTER_LOG"
  for port in 8006 8007 8008 8009 8086; do kill_port "$port"; done
  sleep 5
  for gpu in 0 1 2 3; do
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
      kill -9 "$pid" 2>/dev/null || true
    done
  done
  sleep 4

  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" --port 8086 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.14 --max-model-len "$ORCH_MAX_LEN" --max-num-seqs 64 \
    --served-model-name "$ROUTER_NAME" \
    > logs/orch-gate8k-gpu0-8086.log 2>&1 &
  wait_model http://127.0.0.1:8086/v1 "$ROUTER_NAME" | tee -a "$MASTER_LOG"

  serve_oss() {
    local gpu=$1 port=$2 util=$3 seqs=$4
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -m vllm.entrypoints.openai.api_server \
      --model "$OSS_MODEL" --port "$port" --tensor-parallel-size 1 \
      --gpu-memory-utilization "$util" --max-model-len "$OSS_MAX_LEN" \
      --max-num-seqs "$seqs" --served-model-name gpt-oss-20b \
      > "logs/oss-gate8k-gpu${gpu}-${port}.log" 2>&1 &
    echo "  OSS gpu$gpu :$port pid=$!" | tee -a "$MASTER_LOG"
  }
  serve_oss 0 8006 0.78 16
  wait_model http://127.0.0.1:8006/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
  serve_oss 1 8007 0.92 32
  wait_model http://127.0.0.1:8007/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
  serve_oss 2 8008 0.92 32
  wait_model http://127.0.0.1:8008/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
  serve_oss 3 8009 0.92 32
  wait_model http://127.0.0.1:8009/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
fi

run_mode() {
  local mode=$1
  local log="logs/eval-hintflow-gate-idist8k-${mode}.log"
  echo "=== $(date -Is) mode=$mode ===" | tee -a "$MASTER_LOG" | tee "$log"
  python HintFlow_gate/eval_gate.py \
    --data-file "$LABELS" \
    --out-dir "$OUT" \
    --mode "$mode" \
    --orch-url http://127.0.0.1:8086/v1 \
    --orch-model "$ROUTER_NAME" \
    --solver-urls "$SOLVER_URLS" \
    --solver-model gpt-oss-20b \
    --solver-max-tokens "$TOKENS" \
    --use-threshold "$THRESH" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    2>&1 | tee -a "$log" | tee -a "$MASTER_LOG"
}

# Only gated (user request). Compare to existing HF1@8k baseline / router refs.
run_mode gated

echo "=== $(date -Is) compare ===" | tee -a "$MASTER_LOG"
python3 - <<'PY' | tee -a "$MASTER_LOG"
import json
from pathlib import Path

def pct(x): return f"{100*x:.2f}%"
out = Path("checkpoints/eval_hintflow_gate_idist_1809_8k")
s = json.loads((out / "summary.json").read_text())
print("=== HintFlow_gate IID@8k (gated only) ===")
h = s.get("modes", {}).get("gated", {})
if h:
    print(f"  gated EM={pct(h['em'])}  used_hint={h.get('used_hint_rate',0)*100:.1f}%  err={h.get('n_error')}")

# Pair vs existing HF1@8k baseline (same n=1809 labels)
hf1_path = Path("checkpoints/eval_hintflow_one_idist_1809_8k/hintflow_one.jsonl")
gate_path = out / "gated.jsonl"
if hf1_path.exists() and gate_path.exists():
    base = {r["id"]: r for r in map(json.loads, open(hf1_path))}
    gate = {r["id"]: r for r in map(json.loads, open(gate_path)) if not r.get("error")}
    ids = sorted(set(base) & set(gate))
    rec = harm = 0
    bem = gem = 0
    for i in ids:
        b = int(bool(base[i].get("baseline_em")))
        g = int(bool(gate[i].get("em")))
        bem += b; gem += g
        if g and not b: rec += 1
        if b and not g: harm += 1
    n = len(ids)
    print(f"  paired vs HF1@8k baseline: n={n} bare={pct(bem/n)} gated={pct(gem/n)} "
          f"Δ={(gem-bem)/n*100:+.2f}pp  rec/harm={rec}/{harm}")

refs = [
    ("HF1@8k final", "checkpoints/eval_hintflow_one_idist_1809_8k/summary.json",
     lambda d: d["hintflow_one"]["em"]),
    ("HF1@8k base", "checkpoints/eval_hintflow_one_idist_1809_8k/summary.json",
     lambda d: d["hintflow_one"]["baseline_em"]),
    ("router~8k", "checkpoints/eval_idist_blind_ff_17k_full/summary.json",
     lambda d: d["ff_router"]["em"]),
]
print("=== refs (existing, not re-run) ===")
for tag, path, pick in refs:
    p = Path(path)
    if not p.exists():
        print(f"  {tag}: MISSING"); continue
    print(f"  {tag:14s} EM={pct(pick(json.loads(p.read_text())))}")
PY

echo "=== $(date -Is) DONE ===" | tee -a "$MASTER_LOG"
