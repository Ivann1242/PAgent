#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=${SESSION:-pagent-idist-clean500-overfit30}
MERGED=${MERGED:-checkpoints/blind_ff_sft_qwen3_14b_nothink_clean_overfit30_merged}
LABELS=${LABELS:-checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_clean_train_sample500.jsonl}
OUT=${OUT:-checkpoints/eval_idist_qwen3_14b_nothink_clean_sft_overfit30_train500}
LOG=${LOG:-logs/eval-idist-clean500-overfit30.log}
ROUTER_NAME=${ROUTER_NAME:-qwen3-14b-blind-ff-clean-overfit30}
ROUTER_PORT=${ROUTER_PORT:-8087}

mkdir -p logs "$OUT"
test -f "$MERGED/model.safetensors"
test -f "$LABELS"

tmux new-session -d -s "$SESSION" bash -lc "
set -euo pipefail
cd /home/ivaning/PAgent
: > '$LOG'
echo \"=== \$(date -Is) serve overfit30 router GPU0 :$ROUTER_PORT ===\" | tee -a '$LOG'
CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 python -m vllm.entrypoints.openai.api_server \\
  --model '$MERGED' \\
  --port '$ROUTER_PORT' \\
  --tensor-parallel-size 1 \\
  --gpu-memory-utilization 0.40 \\
  --max-model-len 8192 \\
  --max-num-seqs 32 \\
  --served-model-name '$ROUTER_NAME' \\
  > logs/serve-router-clean-overfit30-8087.log 2>&1 &
router_pid=\$!
trap 'kill \$router_pid 2>/dev/null || true' EXIT

ok=0
for _ in \$(seq 1 120); do
  if curl -sf http://127.0.0.1:$ROUTER_PORT/v1/models | grep -q '$ROUTER_NAME' \\
     && curl -sf http://127.0.0.1:8007/v1/models | grep -q qwen3-14b \\
     && curl -sf http://127.0.0.1:8008/v1/models | grep -q qwen3-14b; then
    ok=1
    break
  fi
  sleep 5
done
if [[ \$ok -ne 1 ]]; then
  echo 'servers not ready' | tee -a '$LOG'
  tail -40 logs/serve-router-clean-overfit30-8087.log | tee -a '$LOG'
  exit 1
fi
echo \"=== \$(date -Is) servers ready; IID clean500 eval ===\" | tee -a '$LOG'

python eval_idist.py \\
  --labels-file '$LABELS' \\
  --out-dir '$OUT' \\
  --router-model '$ROUTER_NAME' \\
  --router-url http://127.0.0.1:$ROUTER_PORT/v1 \\
  --answer-urls http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1 \\
  --modes live_baseline ff_router oracle_hint \\
  --workers 64 \\
  --max-tokens 8192 \\
  --protocol native \\
  2>&1 | tee -a '$LOG'

echo \"=== \$(date -Is) ALL DONE ===\" | tee -a '$LOG'
"

echo "started tmux: $SESSION"
echo "log: $LOG"
echo "out: $OUT"
