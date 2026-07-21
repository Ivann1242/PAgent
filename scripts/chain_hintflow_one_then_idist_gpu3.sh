#!/usr/bin/env bash
# Wait for HintFlow_one 20k eval, then run Blind FF iid (1809) on GPU3 serves only.
# Does NOT start/stop vLLM and does NOT touch other GPUs.
set -euo pipefail
cd /home/ivaning/PAgent
mkdir -p logs

HF_SESSION=${HF_SESSION:-pagent-hintflow-one-orch-eval-gpu3}
HF_OUT=${HF_OUT:-checkpoints/eval_hintflow_one_orch_blindff17k_gpu3_128_20k}
HF_LOG=${HF_LOG:-logs/eval-hintflow-one-orch-gpu3.log}
IDIST_LOG=${IDIST_LOG:-logs/eval-idist-blindff17k-20k-gpu3.log}
IDIST_OUT=${IDIST_OUT:-checkpoints/eval_idist_blind_ff_17k_full_20k}
IDIST_LABELS=${IDIST_LABELS:-checkpoints/blind_hint_17k/oracle_labels_dedup.jsonl}
TOKENS=${TOKENS:-20000}
WORKERS=${WORKERS:-1}

echo "=== $(date -Is) wait for HintFlow_one session/out ===" | tee -a "$IDIST_LOG"

# Prefer summary.json; also accept tmux session exit.
while true; do
  if [[ -f "$HF_OUT/summary.json" ]]; then
    echo "HintFlow_one summary ready: $HF_OUT/summary.json" | tee -a "$IDIST_LOG"
    break
  fi
  if ! tmux has-session -t "$HF_SESSION" 2>/dev/null; then
    if [[ -f "$HF_OUT/summary.json" ]]; then
      break
    fi
    echo "ERROR: tmux $HF_SESSION gone but no summary at $HF_OUT" | tee -a "$IDIST_LOG"
    exit 1
  fi
  sleep 60
done

echo "=== HintFlow_one summary ===" | tee -a "$IDIST_LOG"
python -c "import json; print(json.dumps(json.load(open('$HF_OUT/summary.json')), indent=2))" | tee -a "$IDIST_LOG"

# Health check existing GPU3 serves (do not restart).
if ! curl -sf "http://127.0.0.1:8006/v1/models" | grep -q "gpt-oss-20b"; then
  echo "ERROR: required serve missing: gpt-oss-20b at :8006" | tee -a "$IDIST_LOG"
  exit 1
fi
if ! curl -sf "http://127.0.0.1:8086/v1/models" | grep -q "qwen3-4b-blind-ff-17k"; then
  echo "ERROR: required serve missing: qwen3-4b-blind-ff-17k at :8086" | tee -a "$IDIST_LOG"
  exit 1
fi

echo "=== $(date -Is) start iid eval n≈1809 tokens=$TOKENS workers=$WORKERS ===" | tee -a "$IDIST_LOG"
python eval_idist.py \
  --labels-file "$IDIST_LABELS" \
  --out-dir "$IDIST_OUT" \
  --router-model qwen3-4b-blind-ff-17k \
  --router-url http://127.0.0.1:8086/v1 \
  --workers "$WORKERS" \
  --max-tokens "$TOKENS" \
  --modes live_baseline ff_router oracle_hint \
  2>&1 | tee -a "$IDIST_LOG"

echo "=== $(date -Is) iid DONE ===" | tee -a "$IDIST_LOG"
python - <<PY | tee -a "$IDIST_LOG"
import json
s = json.load(open("$IDIST_OUT/summary.json"))
print(
    "n=", s["meta"]["n_questions"],
    "max_tokens=", s["meta"].get("max_tokens"),
    "baseline=", f"{s['live_baseline']['em']*100:.2f}%",
    "router=", f"{s['ff_router']['em']*100:.2f}%",
    "oracle_hint=", f"{s['oracle_hint']['em']*100:.2f}%",
)
PY
