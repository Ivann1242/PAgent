#!/usr/bin/env bash
# HintFlow_one ×4 sequential repeats on existing GPU3 serves (no restart, GPU3 only).
set -euo pipefail
cd /home/ivaning/PAgent

REPEATS=${REPEATS:-4}
BASE_SEED=${BASE_SEED:-41}
WORKERS=${WORKERS:-1}
TOKENS=${TOKENS:-20000}
ORCH_MODEL=${ORCH_MODEL:-qwen3-4b-blind-ff-17k}
SESSION=${SESSION:-pagent-hintflow-one-repeat4-gpu3}
ROOT_OUT=${ROOT_OUT:-checkpoints/eval_hintflow_one_blindff17k_repeat4_20k}
MASTER_LOG=${MASTER_LOG:-logs/eval-hintflow-one-repeat4-20k.log}
# Optional: reuse an existing seed-41 run dir to skip the first repeat.
REUSE_SEED41=${REUSE_SEED41:-checkpoints/eval_hintflow_one_orch_blindff17k_gpu3_128_20k}

if [[ ${RUN_INSIDE_TMUX:-0} != 1 ]]; then
  mkdir -p logs "$ROOT_OUT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "RUN_INSIDE_TMUX=1 REPEATS=$REPEATS BASE_SEED=$BASE_SEED WORKERS=$WORKERS TOKENS=$TOKENS ORCH_MODEL=$ORCH_MODEL ROOT_OUT=$ROOT_OUT MASTER_LOG=$MASTER_LOG REUSE_SEED41=$REUSE_SEED41 bash '$0'"
  echo "started tmux: $SESSION"
  echo "  attach: tmux attach -t $SESSION"
  echo "  log:    tail -f $MASTER_LOG"
  echo "  out:    $ROOT_OUT/repeat_summary.json"
  exit 0
fi

mkdir -p logs "$ROOT_OUT"
exec > >(tee "$MASTER_LOG") 2>&1

echo "=== $(date -Is) HintFlow_one repeat${REPEATS} tokens=$TOKENS workers=$WORKERS orch=$ORCH_MODEL ==="
curl -sf http://127.0.0.1:8006/v1/models | grep -q gpt-oss-20b
curl -sf http://127.0.0.1:8086/v1/models | grep -q "$ORCH_MODEL"

for ((i = 1; i <= REPEATS; i++)); do
  seed=$((BASE_SEED + i - 1))
  out="$ROOT_OUT/run_${i}_seed_${seed}"
  echo "=== repeat $i/$REPEATS seed=$seed ==="

  if [[ "$i" -eq 1 && -n "${REUSE_SEED41}" && -f "${REUSE_SEED41}/summary.json" ]]; then
    echo "reuse existing seed41: $REUSE_SEED41 -> $out"
    mkdir -p "$out"
    cp -f "$REUSE_SEED41/summary.json" "$out/summary.json"
    if [[ -f "$REUSE_SEED41/hintflow_one.jsonl" ]]; then
      cp -f "$REUSE_SEED41/hintflow_one.jsonl" "$out/hintflow_one.jsonl"
    fi
    continue
  fi

  python HintFlow_one/eval_one.py \
    --data-file data/DAPO-Math.parquet \
    --limit 128 \
    --workers "$WORKERS" \
    --solver-urls http://127.0.0.1:8006/v1 \
    --orch-url http://127.0.0.1:8086/v1 \
    --orch-model "$ORCH_MODEL" \
    --solver-model gpt-oss-20b \
    --solver-max-tokens "$TOKENS" \
    --selector-mode orch \
    --replace-threshold 0.90 \
    --seed "$seed" \
    --out-dir "$out"
done

python - "$ROOT_OUT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = []
for path in sorted(root.glob("run_*/summary.json")):
    runs.append(json.loads(path.read_text()))

def g(run, key):
    return run["hintflow_one"][key]

ems = [g(r, "em") for r in runs]
base = [g(r, "baseline_em") for r in runs]
chal = [g(r, "challenger_em") for r in runs]
deltas = [g(r, "paired_delta") for r in runs]
summary = {
    "n_runs": len(runs),
    "solver_max_tokens": runs[0]["meta"].get("solver_max_tokens") if runs else None,
    "orch_model": runs[0]["meta"].get("orch_model") if runs else None,
    "seeds": [r["meta"]["seed"] for r in runs],
    "em": ems,
    "em_mean": statistics.mean(ems) if ems else 0.0,
    "em_stdev": statistics.stdev(ems) if len(ems) > 1 else 0.0,
    "baseline_em": base,
    "baseline_em_mean": statistics.mean(base) if base else 0.0,
    "challenger_em": chal,
    "challenger_em_mean": statistics.mean(chal) if chal else 0.0,
    "paired_delta": deltas,
    "paired_delta_mean": statistics.mean(deltas) if deltas else 0.0,
    "total_recovered": sum(g(r, "recovered") for r in runs),
    "total_harmed": sum(g(r, "harmed") for r in runs),
    "retention_mean": statistics.mean(
        [g(r, "baseline_correct_retention") for r in runs]
    ) if runs else 0.0,
    "runs": [
        {
            "seed": r["meta"]["seed"],
            "em": g(r, "em"),
            "baseline_em": g(r, "baseline_em"),
            "challenger_em": g(r, "challenger_em"),
            "paired_delta": g(r, "paired_delta"),
            "recovered": g(r, "recovered"),
            "harmed": g(r, "harmed"),
            "replace_count": g(r, "replace_count"),
        }
        for r in runs
    ],
}
(root / "repeat_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "=== $(date -Is) DONE ==="
echo "out: $ROOT_OUT/repeat_summary.json"
