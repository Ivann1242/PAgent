#!/usr/bin/env bash
# HF1 ablations on IID 1809 @8k with max 4-GPU throughput.
# A) challenger = resampled bare baseline @ T=0.7
# B) challenger = fixed CoT (no trained router)
# Layout: GPU0 = orch(selector) + OSS; GPU1/2/3 = OSS  → 4 solvers + selector
set -euo pipefail
cd /home/ivaning/PAgent

MASTER_LOG=${MASTER_LOG:-logs/hf1-ablation-iid8k-4gpu.log}
LABELS=checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl
MERGED=${MERGED:-checkpoints/blind_ff_sft_17k_merged}
ROUTER_NAME=${ROUTER_NAME:-qwen3-4b-blind-ff-17k}
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}
TOKENS=8192
OSS_MAX_LEN=16384
ORCH_MAX_LEN=8192
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_SEQS_SHARED=${MAX_NUM_SEQS_SHARED:-16}
WORKERS=${WORKERS:-80}
SEED=41
CHAL_TEMP=${CHAL_TEMP:-0.7}

OUT_A=checkpoints/eval_hintflow_one_idist_1809_8k_ablate_resample
OUT_B=checkpoints/eval_hintflow_one_idist_1809_8k_ablate_fixedcot
LOG_A=logs/eval-hf1-ablate-resample-iid8k.log
LOG_B=logs/eval-hf1-ablate-fixedcot-iid8k.log

mkdir -p logs "$OUT_A" "$OUT_B"

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

{
  echo "=== $(date -Is) HF1 ablation IID@8k start ==="
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
  free -h | head -2
} | tee "$MASTER_LOG"

echo "=== $(date -Is) stop old vLLM ===" | tee -a "$MASTER_LOG"
for port in 8006 8007 8008 8009 8086; do kill_port "$port"; done
sleep 5
for gpu in 0 1 2 3; do
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
  done
done
sleep 4

echo "=== $(date -Is) serve orch on GPU0 :8086 ===" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --port 8086 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.14 --max-model-len "$ORCH_MAX_LEN" --max-num-seqs 64 \
  --served-model-name "$ROUTER_NAME" \
  > logs/orch-ablate8k-gpu0-8086.log 2>&1 &
echo "  orch pid=$!" | tee -a "$MASTER_LOG"
wait_model http://127.0.0.1:8086/v1 "$ROUTER_NAME" | tee -a "$MASTER_LOG"

serve_oss() {
  local gpu=$1 port=$2 util=$3 seqs=$4
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$OSS_MODEL" --port "$port" --tensor-parallel-size 1 \
    --gpu-memory-utilization "$util" --max-model-len "$OSS_MAX_LEN" \
    --max-num-seqs "$seqs" --served-model-name gpt-oss-20b \
    > "logs/oss-ablate8k-gpu${gpu}-${port}.log" 2>&1 &
  echo "  OSS gpu$gpu :$port pid=$! util=$util seqs=$seqs" | tee -a "$MASTER_LOG"
}

echo "=== $(date -Is) serve 4x OSS (GPU0 shared, GPU1-3 full) ===" | tee -a "$MASTER_LOG"
# GPU0 shares with orch → lower util / fewer concurrent seqs
serve_oss 0 8006 0.78 "$MAX_NUM_SEQS_SHARED"
wait_model http://127.0.0.1:8006/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
serve_oss 1 8007 0.92 "$MAX_NUM_SEQS"
wait_model http://127.0.0.1:8007/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
serve_oss 2 8008 0.92 "$MAX_NUM_SEQS"
wait_model http://127.0.0.1:8008/v1 gpt-oss-20b | tee -a "$MASTER_LOG"
serve_oss 3 8009 0.92 "$MAX_NUM_SEQS"
wait_model http://127.0.0.1:8009/v1 gpt-oss-20b | tee -a "$MASTER_LOG"

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$MASTER_LOG"

SOLVER_URLS=http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1

run_ablate() {
  local mode=$1 out=$2 log=$3 extra=("${@:4}")
  echo "=== $(date -Is) ablation mode=$mode out=$out ===" | tee -a "$MASTER_LOG" | tee "$log"
  python HintFlow_one/eval_one.py \
    --data-file "$LABELS" \
    --out-dir "$out" \
    --orch-url http://127.0.0.1:8086/v1 \
    --orch-model "$ROUTER_NAME" \
    --solver-urls "$SOLVER_URLS" \
    --solver-model gpt-oss-20b \
    --solver-max-tokens "$TOKENS" \
    --selector-mode orch \
    --replace-threshold 0.90 \
    --challenger-mode "$mode" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    "${extra[@]}" \
    2>&1 | tee -a "$log" | tee -a "$MASTER_LOG"
  python3 - <<PY | tee -a "$MASTER_LOG" | tee -a "$log"
import json
s=json.load(open("$out/summary.json"))
h=s["hintflow_one"]; m=s["meta"]
print(f"DONE {m.get('challenger_mode')}: EM={h['em']*100:.2f}% base={h['baseline_em']*100:.2f}% "
      f"chal={h['challenger_em']*100:.2f}% d={h['paired_delta']*100:+.2f}pp "
      f"rec={h['recovered']} harm={h['harmed']} replace={h['replace_count']} "
      f"err={h['n_error']} t={m['elapsed_sec']}s")
PY
}

# A: resampled baseline @ T=0.7
run_ablate resample_baseline "$OUT_A" "$LOG_A" --challenger-temperature "$CHAL_TEMP"

# B: fixed CoT @ T=0 (greedy; diversity comes from prompt, not sampling)
run_ablate fixed_cot "$OUT_B" "$LOG_B" --challenger-temperature 0.0

echo "=== $(date -Is) compare vs trained HF1@8k ===" | tee -a "$MASTER_LOG"
python3 - <<'PY' | tee -a "$MASTER_LOG"
import json
from pathlib import Path
rows=[]
for tag,p in [
    ("trained HF1", "checkpoints/eval_hintflow_one_idist_1809_8k/summary.json"),
    ("ablate resample T=0.7", "checkpoints/eval_hintflow_one_idist_1809_8k_ablate_resample/summary.json"),
    ("ablate fixed CoT", "checkpoints/eval_hintflow_one_idist_1809_8k_ablate_fixedcot/summary.json"),
]:
    if not Path(p).exists():
        print(f"{tag}: MISSING"); continue
    h=json.load(open(p))["hintflow_one"]
    print(f"{tag:22s}  EM={h['em']*100:5.2f}%  base={h['baseline_em']*100:5.2f}%  "
          f"chal={h['challenger_em']*100:5.2f}%  Δ={h['paired_delta']*100:+6.2f}pp  "
          f"rec/harm={h['recovered']}/{h['harmed']}")
PY

echo "=== $(date -Is) ALL ABLATIONS DONE ===" | tee -a "$MASTER_LOG"
