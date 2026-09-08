#!/usr/bin/env bash
set -euo pipefail
cd /home/ivaning/PAgent

SESSION=${SESSION:-pagent-hf1-qwen-clean128-det-ablate}
DATA=${DATA:-checkpoints/blind_hint_17k_qwen3_14b_nothink/oracle_labels_clean_train_sample128.jsonl}
ROOT=${ROOT:-checkpoints/eval_hf1_qwen_clean128_deterministic_ablation}
LOG=${LOG:-logs/eval-hf1-qwen-clean128-deterministic-ablation.log}
SOLVER_URLS=${SOLVER_URLS:-http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1}
SOLVER_MODEL=${SOLVER_MODEL:-qwen3-14b}
WORKERS=${WORKERS:-32}
TOKENS=${TOKENS:-8192}
SEED=${SEED:-41}

mkdir -p logs "$ROOT"
test -f "$DATA"

tmux new-session -d -s "$SESSION" bash -lc "
set -euo pipefail
cd /home/ivaning/PAgent
: > '$LOG'

for port in 8007 8008; do
  curl -sf \"http://127.0.0.1:\$port/v1/models\" | grep -q '$SOLVER_MODEL'
done

run_one() {
  tag=\$1
  mode=\$2
  echo \"=== \$(date -Is) \$tag mode=\$mode T=0 ===\" | tee -a '$LOG'
  python HintFlow_one/eval_one.py \\
    --data-file '$DATA' \\
    --out-dir '$ROOT'/\"\$tag\" \\
    --solver-urls '$SOLVER_URLS' \\
    --solver-model '$SOLVER_MODEL' \\
    --solver-max-tokens '$TOKENS' \\
    --selector-mode replace \\
    --challenger-mode \"\$mode\" \\
    --challenger-temperature 0.0 \\
    --seed '$SEED' \\
    --workers '$WORKERS' \\
    2>&1 | tee -a '$LOG'
}

run_one baseline_t0_run1 resample_baseline
run_one baseline_t0_run2 resample_baseline
run_one fixed_cot_t0 fixed_cot

python3 - <<'PY' | tee -a '$LOG'
import json
from pathlib import Path

root = Path('$ROOT')
print('=== FINAL ===')
for tag in ('baseline_t0_run1', 'baseline_t0_run2', 'fixed_cot_t0'):
    h = json.loads((root / tag / 'summary.json').read_text())['hintflow_one']
    print(
        f\"{tag}: base={100*h['baseline_em']:.2f}% \"
        f\"challenger={100*h['challenger_em']:.2f}% \"
        f\"final={100*h['em']:.2f}% errors={h['n_error']}\"
    )
print(f\"oracle reference: 100.00% ({len((root / 'baseline_t0_run1' / 'hintflow_one.jsonl').read_text().splitlines())}/128 expected rows)\")
PY

echo \"=== \$(date -Is) ALL DONE ===\" | tee -a '$LOG'
"

echo "started tmux: $SESSION"
echo "log: $LOG"
echo "out: $ROOT"
