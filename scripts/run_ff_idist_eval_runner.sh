#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent

LOG_DIR=/home/ivaning/PAgent/logs
LOG_FILE="$LOG_DIR/ff-idist-eval.log"
PORT=8086
GPU=1
IDIST_LABELS=checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl
WORKERS=32

: > "$LOG_FILE"

run_one_model() {
  local name="$1"
  local merged="$2"
  local model_name="$3"
  local out_root="$4"

  pkill -f "api_server.*--port ${PORT}" 2>/dev/null || true
  sleep 5

  echo "=== serve ${model_name} ===" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --model "$merged" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.25 \
    --max-model-len 8192 \
    --served-model-name "$model_name" \
    > "$LOG_DIR/${name}-serve.log" 2>&1 &
  local pid=$!

  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" | grep -q "$model_name"; then
      echo "serve ready: $model_name" | tee -a "$LOG_FILE"
      break
    fi
    sleep 5
  done

  echo "=== idist eval: ${name} ===" | tee -a "$LOG_FILE"
  python eval_idist.py \
    --labels-file "$IDIST_LABELS" \
    --out-dir "$out_root" \
    --router-model "$model_name" \
    --router-url "http://127.0.0.1:${PORT}/v1" \
    --workers "$WORKERS" \
    2>&1 | tee -a "$LOG_FILE"

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

echo "idist questions: $(python -c "from pathlib import Path; from eval_idist import load_idist_rows; print(len(load_idist_rows(Path('$IDIST_LABELS'))))")" | tee -a "$LOG_FILE"

run_one_model full \
  checkpoints/blind_ff_sft_17k_merged \
  qwen3-4b-blind-ff-17k \
  checkpoints/eval_idist_blind_ff_17k_full

run_one_model dedup \
  checkpoints/blind_ff_sft_17k_dedup_merged \
  qwen3-4b-blind-ff-17k-dedup \
  checkpoints/eval_idist_blind_ff_17k_dedup

echo "=== compare ===" | tee -a "$LOG_FILE"
python - <<'PY' | tee -a "$LOG_FILE"
import json
from pathlib import Path
for name in ["full", "dedup"]:
    p = Path(f"checkpoints/eval_idist_blind_ff_17k_{name}/summary.json")
    s = json.loads(p.read_text())
    print(
        name,
        "n=", s["meta"]["n_questions"],
        "baseline=", f"{s['live_baseline']['em']*100:.2f}%",
        "router=", f"{s['ff_router']['em']*100:.2f}%",
        "oracle_hint=", f"{s['oracle_hint']['em']*100:.2f}%",
        "delta=", f"{s['ff_router_vs_live_baseline']*100:+.2f}pp",
    )
PY
echo "=== DONE ===" | tee -a "$LOG_FILE"
