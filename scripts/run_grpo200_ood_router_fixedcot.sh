#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent

ROOT=checkpoints/eval_grpo200_ood_router_fixedcot_qwen14b_8k
LOG=logs/eval-grpo200-ood-router-fixedcot-qwen14b-8k.log
ROUTER=qwen3-4b-blind-ff-grpo-clean-iid-vp-step200
SOLVERS=http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1
mkdir -p "$ROOT" logs

run_one() {
  local tag=$1 data=$2 limit=$3 mode=$4
  local out="$ROOT/${tag}_${mode}"
  local limit_args=()
  if [[ -n "$limit" ]]; then limit_args=(--limit "$limit"); fi
  python HintFlow_one/eval_one.py \
    --data-file "$data" \
    "${limit_args[@]}" \
    --out-dir "$out" \
    --orch-url http://127.0.0.1:8086/v1 \
    --orch-model "$ROUTER" \
    --solver-urls "$SOLVERS" \
    --solver-model qwen3-14b \
    --solver-max-tokens 8192 \
    --selector-mode replace \
    --challenger-mode "$mode" \
    --challenger-temperature 0.0 \
    --seed 41 \
    --workers 32
}

{
  echo "=== $(date -Is) GRPO200 OOD: baseline/router/fixed-CoT; Qwen3-14B@8k ==="
  for spec in \
    'aime24|data/aime24.jsonl|' \
    'dapo128|data/DAPO-Math.parquet|128'; do
    IFS='|' read -r tag data limit <<<"$spec"
    echo "=== $(date -Is) $tag router ==="
    run_one "$tag" "$data" "$limit" blind_ff
    echo "=== $(date -Is) $tag fixed-CoT ==="
    run_one "$tag" "$data" "$limit" fixed_cot
  done
  python3 - <<'PY'
import json
from pathlib import Path
root = Path("checkpoints/eval_grpo200_ood_router_fixedcot_qwen14b_8k")
print("\n=== FINAL ===")
for tag in ("aime24", "dapo128"):
    r = json.loads((root / f"{tag}_blind_ff/summary.json").read_text())["hintflow_one"]
    f = json.loads((root / f"{tag}_fixed_cot/summary.json").read_text())["hintflow_one"]
    print(
        f"{tag}: baseline={100*r['baseline_em']:.2f}% "
        f"router={100*r['challenger_em']:.2f}% ({100*(r['challenger_em']-r['baseline_em']):+.2f}pp) "
        f"fixedcot={100*f['challenger_em']:.2f}% ({100*(f['challenger_em']-f['baseline_em']):+.2f}pp) "
        f"baseline_check={100*f['baseline_em']:.2f}% "
        f"errors={r['n_error']}/{f['n_error']}"
    )
PY
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee "$LOG"
