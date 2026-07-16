#!/usr/bin/env bash
# Four sequential retained-V2 repeats on GPU3, with reproducible distinct seeds.
set -euo pipefail

cd /home/ivaning/PAgent

REPEATS=${REPEATS:-4}
BASE_SEED=${BASE_SEED:-41}
WORKERS=${WORKERS:-4}
SOLVER_MAX_TOKENS=${SOLVER_MAX_TOKENS:-4096}
SESSION=${SESSION:-pagent-hintflow-retained-fixed-repeat4}
ROOT_OUT=${ROOT_OUT:-checkpoints/eval_hintflow_retained_fixed_repeat4}
MASTER_LOG=${MASTER_LOG:-logs/eval-hintflow-retained-fixed-repeat4.log}

if [[ ${RUN_INSIDE_TMUX:-0} != 1 ]]; then
  mkdir -p logs "$ROOT_OUT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "RUN_INSIDE_TMUX=1 REPEATS=$REPEATS BASE_SEED=$BASE_SEED WORKERS=$WORKERS SOLVER_MAX_TOKENS=$SOLVER_MAX_TOKENS ROOT_OUT=$ROOT_OUT MASTER_LOG=$MASTER_LOG bash '$0'"
  echo "started tmux: $SESSION"
  echo "  attach: tmux attach -t $SESSION"
  echo "  log:    tail -f $MASTER_LOG"
  echo "  out:    $ROOT_OUT/repeat_summary.json"
  exit 0
fi

mkdir -p logs "$ROOT_OUT"
exec > >(tee "$MASTER_LOG") 2>&1

for ((i = 1; i <= REPEATS; i++)); do
  seed=$((BASE_SEED + i - 1))
  out="$ROOT_OUT/run_${i}_seed_${seed}"
  echo "=== repeat $i/$REPEATS seed=$seed ==="
  python HintFlow/eval_hintflow.py \
    --data-file data/DAPO-Math.parquet \
    --limit 128 \
    --workers "$WORKERS" \
    --solver-urls http://127.0.0.1:8006/v1 \
    --orch-url http://127.0.0.1:8086/v1 \
    --orch-model qwen3-4b \
    --solver-model gpt-oss-20b \
    --runtime-mode retained \
    --orch-temperature 0 \
    --solver-max-tokens "$SOLVER_MAX_TOKENS" \
    --seed "$seed" \
    --skip-baseline \
    --out-dir "$out"
done

python - "$ROOT_OUT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = [
    json.loads(path.read_text())
    for path in sorted(root.glob("run_*/summary.json"))
]
ems = [run["hintflow"]["em"] for run in runs]
baseline_ems = [run["hintflow"]["internal_baseline_em"] for run in runs]
summary = {
    "n_runs": len(runs),
    "em": ems,
    "em_mean": statistics.mean(ems),
    "em_stdev": statistics.stdev(ems) if len(ems) > 1 else 0.0,
    "internal_baseline_em": baseline_ems,
    "internal_baseline_em_mean": statistics.mean(baseline_ems),
    "paired_delta": [
        em - baseline for em, baseline in zip(ems, baseline_ems)
    ],
    "total_recovered": sum(
        run["hintflow"]["internal_recovered"] for run in runs
    ),
    "total_harmed": sum(
        run["hintflow"]["internal_harmed"] for run in runs
    ),
    "runs": runs,
}
(root / "repeat_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
PY

echo "=== ALL $REPEATS REPEATS DONE ==="
