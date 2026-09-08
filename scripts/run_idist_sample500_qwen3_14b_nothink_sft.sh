#!/usr/bin/env bash
# Quick idist smoke: 500 random train-split questions for Qwen3-14B-nothink SFT router.
# Serves router on GPU0 + 3x solvers on GPU1-3, then baseline / ff_router / oracle_hint.
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=${SESSION:-pagent-idist-sample500-nothink-sft}
LABELS=${LABELS:-checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_train_sample500.jsonl}
OUT=${OUT:-checkpoints/eval_idist_qwen3_14b_nothink_sft_train500}
MERGED=${MERGED:-checkpoints/blind_ff_sft_qwen3_14b_nothink_merged}
ROUTER_NAME=${ROUTER_NAME:-qwen3-4b-blind-ff-qwen3-14b-nothink}
ROUTER_PORT=${ROUTER_PORT:-8086}
LOG=${LOG:-logs/eval-idist-qwen3-14b-nothink-sft-train500.log}
TOKENS=${TOKENS:-8192}
WORKERS=${WORKERS:-96}
SOLVER=${SOLVER_MODEL_PATH:-/home/ivaning/models/Qwen3-14B}
SERVED=${SOLVER_SERVED_NAME:-qwen3-14b}
MAX_LEN=${SOLVER_MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${SOLVER_MAX_NUM_SEQS:-48}

mkdir -p logs "$OUT"

if [[ ! -f "$LABELS" ]]; then
  echo "missing labels: $LABELS" >&2
  exit 1
fi
if [[ ! -d "$MERGED" ]]; then
  echo "missing merged router: $MERGED" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stopping existing tmux session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" bash -lc "
set -euo pipefail
cd /home/ivaning/PAgent
: > '$LOG'

echo \"=== \$(date -Is) kill old vLLM ===\" | tee -a '$LOG'
pids=\$(ps aux | awk '/vllm.entrypoints.openai.api_server/ && !/awk/ {print \$2}')
if [[ -n \"\${pids}\" ]]; then
  kill \${pids} || true
  sleep 8
fi

echo \"=== \$(date -Is) serve router GPU0 :$ROUTER_PORT ===\" | tee -a '$LOG'
CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model '$MERGED' \
  --port $ROUTER_PORT \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.22 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --served-model-name '$ROUTER_NAME' \
  > logs/serve-router-nothink-sft-8086.log 2>&1 &

echo \"=== \$(date -Is) serve solvers GPU1-3 :8007-8009 ===\" | tee -a '$LOG'
for pair in 1:8007 2:8008 3:8009; do
  gpu=\${pair%%:*}
  port=\${pair##*:}
  CUDA_VISIBLE_DEVICES=\$gpu VLLM_BATCH_INVARIANT=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model '$SOLVER' \
    --port \$port \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len $MAX_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --seed 0 \
    --served-model-name '$SERVED' \
    > logs/serve-qwen3-14b-\${port}.log 2>&1 &
done

echo \"=== \$(date -Is) wait for servers ===\" | tee -a '$LOG'
ok=0
for _ in \$(seq 1 90); do
  if curl -sf http://127.0.0.1:$ROUTER_PORT/v1/models | grep -q '$ROUTER_NAME' \\
     && curl -sf http://127.0.0.1:8007/v1/models | grep -q '$SERVED' \\
     && curl -sf http://127.0.0.1:8008/v1/models | grep -q '$SERVED' \\
     && curl -sf http://127.0.0.1:8009/v1/models | grep -q '$SERVED'; then
    ok=1
    break
  fi
  sleep 5
done
if [[ \$ok -ne 1 ]]; then
  echo 'servers not ready' | tee -a '$LOG'
  tail -30 logs/serve-router-nothink-sft-8086.log | tee -a '$LOG' || true
  exit 1
fi
echo 'all servers ready' | tee -a '$LOG'

echo \"=== \$(date -Is) idist eval n=\$(wc -l < '$LABELS') tokens=$TOKENS ===\" | tee -a '$LOG'
python eval_idist.py \\
  --labels-file '$LABELS' \\
  --out-dir '$OUT' \\
  --router-model '$ROUTER_NAME' \\
  --router-url http://127.0.0.1:$ROUTER_PORT/v1 \\
  --answer-urls http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1 \\
  --modes live_baseline ff_router oracle_hint \\
  --workers $WORKERS \\
  --max-tokens $TOKENS \\
  --protocol native \\
  2>&1 | tee -a '$LOG'

echo \"=== \$(date -Is) DONE ===\" | tee -a '$LOG'
python - <<'PY' | tee -a '$LOG'
import json
from pathlib import Path
s = json.loads(Path('$OUT/summary.json').read_text())
b = s['live_baseline']['em']*100
r = s['ff_router']['em']*100
o = s['oracle_hint']['em']*100
print(f\"n={s['meta']['n_questions']}\")
print(f\"baseline={b:.2f}%  router={r:.2f}%  oracle={o:.2f}%  delta={r-b:+.2f}pp\")
PY
"

echo "started tmux: $SESSION"
echo "  log: tail -f $LOG"
echo "  out: $OUT"
echo "  labels: $(wc -l < "$LABELS") from $LABELS"
