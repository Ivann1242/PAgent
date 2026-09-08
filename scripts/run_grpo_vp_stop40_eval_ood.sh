#!/usr/bin/env bash
# Wait for virtual-positive GRPO to finish step 40, stop training, merge,
# then HintFlow_one eval on DAPO-128 / AIME24 / MATH-500 @ fair 20k.
# Layout: GPU0-1 OSS solvers, GPU2 GRPO router. Never touch GPU3.
set -euo pipefail
cd /home/ivaning/PAgent

TARGET_STEP=40
ADAPTER=checkpoints/blind_ff_grpo_17k_virtual_positive_adapter
STATE="$ADAPTER/grpo_train_state.json"
TRAIN_LOG=logs/blind-ff-grpo-17k-vp320-train.log
LOG=logs/eval-grpo-vp-step40-ood.log
MERGED=checkpoints/blind_ff_grpo_17k_virtual_positive_step40_merged
ROUTER_NAME=qwen3-4b-blind-ff-grpo-vp-step40
OSS_MODEL=${OSS_MODEL_PATH:-/home/ivaning/models/gpt-oss-20b}
TOKENS=20000
SEED=41

mkdir -p logs "$MERGED"

wait_model() {
  local url=$1 name=$2
  for i in $(seq 1 180); do
    curl -sf "$url" | grep -q "$name" && { echo "  $url OK"; return 0; }
    sleep 3
  done
  echo "FAIL $url"; return 1
}

kill_port() {
  pkill -f "vllm.entrypoints.openai.api_server.*--port $1" 2>/dev/null || true
}

echo "=== $(date -Is) wait for GRPO step ${TARGET_STEP} then stop+eval ===" | tee "$LOG"

# --- 1) Wait until step 40 is saved (and HF upload finished if present) ---
while true; do
  if [[ -f "$STATE" ]]; then
    cur=$(python3 -c "import json; print(json.load(open('$STATE'))['step'])")
    echo "$(date -Is) state.step=$cur" | tee -a "$LOG"
    if [[ "$cur" -ge "$TARGET_STEP" ]]; then
      # Prefer waiting for HF checkpoint line; don't block forever.
      if grep -q "HF checkpoint step=${TARGET_STEP}" "$TRAIN_LOG" 2>/dev/null; then
        echo "HF checkpoint step=${TARGET_STEP} seen" | tee -a "$LOG"
        break
      fi
      # Fallback: give upload up to ~25 min after state hit 40
      age=$(( $(date +%s) - $(stat -c %Y "$STATE") ))
      if [[ "$age" -gt 1500 ]]; then
        echo "WARN: no HF line after ${age}s; proceeding with local adapter" | tee -a "$LOG"
        break
      fi
    fi
  fi
  sleep 30
done

# --- 2) Stop training (do not touch GPU3) ---
echo "=== $(date -Is) stop ff-train ===" | tee -a "$LOG"
pkill -f 'python run.py ff-train --init-adapter checkpoints/blind_ff_grpo_17k_virtual_positive_adapter' 2>/dev/null || true
sleep 5
# If still alive, harder kill of that exact cmdline
if pgrep -f 'python run.py ff-train --init-adapter checkpoints/blind_ff_grpo_17k_virtual_positive_adapter' >/dev/null; then
  pkill -9 -f 'python run.py ff-train --init-adapter checkpoints/blind_ff_grpo_17k_virtual_positive_adapter' || true
  sleep 3
fi
echo "train stopped; state=$(python3 -c "import json; print(json.load(open('$STATE'))['step'])")" | tee -a "$LOG"
echo "STOPPED_AT_STEP_${TARGET_STEP} $(date -Is)" | tee -a "$TRAIN_LOG" | tee -a "$LOG"

# Free GPU0 policy + shared OSS / reward OSS so we can reconfigure for eval
kill_port 8008
kill_port 8006
kill_port 8007
kill_port 8086
kill_port 8087
sleep 5

# --- 3) Free disk for merge (~8GB safetensors) ---
# Old GRPO step17 merged is obsolete; reclaim if present.
if [[ -f checkpoints/blind_ff_grpo_17k_r3_step17_merged/model.safetensors ]]; then
  echo "freeing old GRPO step17 merged weights" | tee -a "$LOG"
  rm -f checkpoints/blind_ff_grpo_17k_r3_step17_merged/model.safetensors
fi
df -h /home/ivaning | tee -a "$LOG"

# --- 4) Merge step-40 adapter ---
echo "=== $(date -Is) ff-merge step${TARGET_STEP} ===" | tee -a "$LOG"
python run.py ff-merge \
  --adapter-dir "$ADAPTER" \
  --merged-dir "$MERGED" \
  2>&1 | tee -a "$LOG"

# --- 5) Serve: GPU0-1 OSS, GPU2 GRPO router ---
echo "=== $(date -Is) start eval servers ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" --port 8006 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 16 \
  --served-model-name gpt-oss-20b \
  > logs/oss-grpo40-gpu0-8006.log 2>&1 &
echo "  OSS gpu0 :8006 pid=$!" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$OSS_MODEL" --port 8007 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 16 \
  --served-model-name gpt-oss-20b \
  > logs/oss-grpo40-gpu1-8007.log 2>&1 &
echo "  OSS gpu1 :8007 pid=$!" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --port 8086 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 32 \
  --served-model-name "$ROUTER_NAME" \
  > logs/orch-grpo40-gpu2-8086.log 2>&1 &
echo "  router gpu2 :8086 pid=$!" | tee -a "$LOG"

wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b | tee -a "$LOG"
wait_model http://127.0.0.1:8086/v1/models "$ROUTER_NAME" | tee -a "$LOG"

echo "GPU3 apps (should be others only):" | tee -a "$LOG"
nvidia-smi -i 3 --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$LOG"

run_eval() {
  local tag=$1 data=$2 out=$3 limit_arg=$4 workers=$5
  echo "=== $(date -Is) HF1 ${tag} @${TOKENS} ===" | tee -a "$LOG"
  mkdir -p "$out"
  # shellcheck disable=SC2086
  python HintFlow_one/eval_one.py \
    --data-file "$data" \
    $limit_arg \
    --out-dir "$out" \
    --orch-url http://127.0.0.1:8086/v1 \
    --orch-model "$ROUTER_NAME" \
    --solver-urls http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1 \
    --solver-model gpt-oss-20b \
    --solver-max-tokens "$TOKENS" \
    --selector-mode orch \
    --replace-threshold 0.90 \
    --seed "$SEED" \
    --workers "$workers" \
    2>&1 | tee -a "$LOG"
  echo "--- ${tag} summary ---" | tee -a "$LOG"
  cat "$out/summary.json" | tee -a "$LOG"
}

# Order: DAPO-128 (signal) → AIME24 → MATH-500
run_eval "DAPO-128" "data/DAPO-Math.parquet" \
  "checkpoints/eval_hintflow_one_grpo_vp_step40_dapo128_20k" \
  "--limit 128" 16

run_eval "AIME24" "data/aime24.jsonl" \
  "checkpoints/eval_hintflow_one_grpo_vp_step40_aime24_20k" \
  "" 16

run_eval "MATH-500" "data/math500.jsonl" \
  "checkpoints/eval_hintflow_one_grpo_vp_step40_math500_20k" \
  "" 32

# --- 6) Compare vs old SFT17k HF1 ---
python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

def load(p):
    p = Path(p)
    if not p.exists():
        return None
    return json.load(open(p))["hintflow_one"]

def fmt(h):
    if h is None:
        return "MISSING"
    return (
        f"base={100*h['baseline_em']:.1f}% "
        f"chal={100*h['challenger_em']:.1f}% "
        f"HF1={100*h['em']:.1f}% "
        f"d={100*h['paired_delta']:+.1f}pp "
        f"rec={h['recovered']} harm={h['harmed']}"
    )

pairs = [
    ("DAPO-128",
     "checkpoints/eval_hintflow_one_grpo_vp_step40_dapo128_20k/summary.json",
     "checkpoints/eval_hintflow_one_orch_blindff17k_gpu3_128_20k/summary.json"),
    ("AIME24",
     "checkpoints/eval_hintflow_one_grpo_vp_step40_aime24_20k/summary.json",
     "checkpoints/eval_hintflow_one_aime24_20k/summary.json"),
    ("MATH-500",
     "checkpoints/eval_hintflow_one_grpo_vp_step40_math500_20k/summary.json",
     "checkpoints/eval_hintflow_one_math500_20k/summary.json"),
]
print("\n=== GRPO vp step40 vs SFT17k HF1 @20k seed41 ===")
for name, new_p, old_p in pairs:
    print(f"\n[{name}]")
    print("  GRPO@40:", fmt(load(new_p)))
    print("  SFT17k: ", fmt(load(old_p)))
PY

echo "=== $(date -Is) ALL EVAL DONE ===" | tee -a "$LOG"
echo "log: $LOG" | tee -a "$LOG"
# Keep merged weights for inspection; free later if disk tight.
df -h /home/ivaning | tee -a "$LOG"
