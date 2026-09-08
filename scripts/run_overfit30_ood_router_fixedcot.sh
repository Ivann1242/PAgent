#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=${SESSION:-pagent-overfit30-ood}
MERGED=${MERGED:-checkpoints/blind_ff_sft_qwen3_14b_nothink_clean_overfit30_merged}
ROOT=${ROOT:-checkpoints/eval_overfit30_ood_router_fixedcot_qwen14b_8k}
LOG=${LOG:-logs/eval-overfit30-ood-router-fixedcot-qwen14b-8k.log}
ROUTER=${ROUTER:-qwen3-14b-blind-ff-clean-overfit30}
ROUTER_PORT=${ROUTER_PORT:-8087}
SOLVERS=${SOLVERS:-http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1}

mkdir -p logs "$ROOT"
test -f "$MERGED/model.safetensors"
test -f data/aime24.jsonl
test -f data/DAPO-Math.parquet

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
  --served-model-name '$ROUTER' \\
  > logs/serve-router-clean-overfit30-ood-8087.log 2>&1 &
router_pid=\$!
trap 'kill \$router_pid 2>/dev/null || true' EXIT

ok=0
for _ in \$(seq 1 120); do
  if curl -sf http://127.0.0.1:$ROUTER_PORT/v1/models | grep -q '$ROUTER' \\
     && curl -sf http://127.0.0.1:8007/v1/models | grep -q qwen3-14b \\
     && curl -sf http://127.0.0.1:8008/v1/models | grep -q qwen3-14b; then
    ok=1
    break
  fi
  sleep 5
done
if [[ \$ok -ne 1 ]]; then
  echo 'servers not ready' | tee -a '$LOG'
  tail -40 logs/serve-router-clean-overfit30-ood-8087.log | tee -a '$LOG'
  exit 1
fi

run_one() {
  local tag=\$1 data=\$2 limit=\$3 mode=\$4
  local out='$ROOT'/\${tag}_\${mode}
  local limit_args=()
  if [[ -n \"\$limit\" ]]; then limit_args=(--limit \"\$limit\"); fi
  python HintFlow_one/eval_one.py \\
    --data-file \"\$data\" \\
    \"\${limit_args[@]}\" \\
    --out-dir \"\$out\" \\
    --orch-url http://127.0.0.1:$ROUTER_PORT/v1 \\
    --orch-model '$ROUTER' \\
    --solver-urls '$SOLVERS' \\
    --solver-model qwen3-14b \\
    --solver-max-tokens 8192 \\
    --selector-mode replace \\
    --challenger-mode \"\$mode\" \\
    --challenger-temperature 0.0 \\
    --seed 41 \\
    --workers 32
}

echo \"=== \$(date -Is) overfit30 OOD: baseline/router/fixed-CoT; Qwen3-14B@8k ===\" | tee -a '$LOG'
for spec in \\
  'aime24|data/aime24.jsonl|' \\
  'dapo128|data/DAPO-Math.parquet|128'; do
  IFS='|' read -r tag data limit <<<\"\$spec\"
  echo \"=== \$(date -Is) \$tag router ===\" | tee -a '$LOG'
  run_one \"\$tag\" \"\$data\" \"\$limit\" blind_ff 2>&1 | tee -a '$LOG'
  echo \"=== \$(date -Is) \$tag fixed-CoT ===\" | tee -a '$LOG'
  run_one \"\$tag\" \"\$data\" \"\$limit\" fixed_cot 2>&1 | tee -a '$LOG'
done
python3 - <<'PY' 2>&1 | tee -a '$LOG'
import json
from pathlib import Path
root = Path('checkpoints/eval_overfit30_ood_router_fixedcot_qwen14b_8k')
print('\n=== FINAL ===')
for tag in ('aime24', 'dapo128'):
    r = json.loads((root / f'{tag}_blind_ff/summary.json').read_text())['hintflow_one']
    f = json.loads((root / f'{tag}_fixed_cot/summary.json').read_text())['hintflow_one']
    print(
        f\"{tag}: baseline={100*r['baseline_em']:.2f}% \"
        f\"router={100*r['challenger_em']:.2f}% ({100*(r['challenger_em']-r['baseline_em']):+.2f}pp) \"
        f\"fixedcot={100*f['challenger_em']:.2f}% ({100*(f['challenger_em']-f['baseline_em']):+.2f}pp) \"
        f\"baseline_check={100*f['baseline_em']:.2f}% \"
        f\"errors={r['n_error']}/{f['n_error']}\"
    )
PY
echo \"=== \$(date -Is) ALL DONE ===\" | tee -a '$LOG'
"

echo "started tmux: $SESSION"
echo "log: $LOG"
echo "out: $ROOT"
