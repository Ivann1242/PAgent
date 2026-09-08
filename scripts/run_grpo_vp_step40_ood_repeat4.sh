#!/usr/bin/env bash
# HintFlow_one ×4 (seeds 41-44) on DAPO-128 / AIME24 / MATH-500 @20k
# for GRPO virtual-positive step40. Reuses existing seed41 runs.
# Expects servers already up: :8006/:8007 OSS, :8086 GRPO router. GPU3 untouched.
set -euo pipefail
cd /home/ivaning/PAgent

REPEATS=${REPEATS:-4}
BASE_SEED=${BASE_SEED:-41}
TOKENS=${TOKENS:-20000}
ROUTER_NAME=${ROUTER_NAME:-qwen3-4b-blind-ff-grpo-vp-step40}
SESSION=${SESSION:-pagent-grpo40-ood-repeat4}
MASTER_LOG=${MASTER_LOG:-logs/eval-grpo-vp-step40-ood-repeat4.log}
ROOT=${ROOT:-checkpoints/eval_hintflow_one_grpo_vp_step40_repeat4_20k}

if [[ ${RUN_INSIDE_TMUX:-0} != 1 ]]; then
  mkdir -p logs "$ROOT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "RUN_INSIDE_TMUX=1 bash '$0'"
  echo "started tmux: $SESSION"
  echo "  attach: tmux attach -t $SESSION"
  echo "  log:    tail -f $MASTER_LOG"
  exit 0
fi

mkdir -p logs "$ROOT"
exec > >(tee "$MASTER_LOG") 2>&1

wait_model() {
  local url=$1 name=$2
  for i in $(seq 1 60); do
    curl -sf "$url" | grep -q "$name" && return 0
    sleep 2
  done
  echo "FAIL $url missing $name"; return 1
}

echo "=== $(date -Is) GRPO vp step40 OOD repeat${REPEATS} @${TOKENS} ==="
wait_model http://127.0.0.1:8006/v1/models gpt-oss-20b
wait_model http://127.0.0.1:8007/v1/models gpt-oss-20b
wait_model http://127.0.0.1:8086/v1/models "$ROUTER_NAME"
echo "GPU3 apps (should be others only):"
nvidia-smi -i 3 --query-compute-apps=pid,process_name,used_memory --format=csv || true

# tag|data|limit_arg|workers|reuse_seed41_dir
DATASETS=(
  "dapo128|data/DAPO-Math.parquet|--limit 128|16|checkpoints/eval_hintflow_one_grpo_vp_step40_dapo128_20k"
  "aime24|data/aime24.jsonl||16|checkpoints/eval_hintflow_one_grpo_vp_step40_aime24_20k"
  "math500|data/math500.jsonl||32|checkpoints/eval_hintflow_one_grpo_vp_step40_math500_20k"
)

aggregate() {
  local root=$1
  python3 - "$root" <<'PY'
import json, statistics, sys
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
    "baseline_em_stdev": statistics.stdev(base) if len(base) > 1 else 0.0,
    "challenger_em": chal,
    "challenger_em_mean": statistics.mean(chal) if chal else 0.0,
    "challenger_em_stdev": statistics.stdev(chal) if len(chal) > 1 else 0.0,
    "paired_delta": deltas,
    "paired_delta_mean": statistics.mean(deltas) if deltas else 0.0,
    "total_recovered": sum(g(r, "recovered") for r in runs),
    "total_harmed": sum(g(r, "harmed") for r in runs),
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
}

for spec in "${DATASETS[@]}"; do
  IFS='|' read -r tag data limit_arg workers reuse <<<"$spec"
  out_root="$ROOT/$tag"
  mkdir -p "$out_root"
  echo "=== $(date -Is) dataset=$tag workers=$workers ==="

  for ((i = 1; i <= REPEATS; i++)); do
    seed=$((BASE_SEED + i - 1))
    out="$out_root/run_${i}_seed_${seed}"
    echo "=== $tag repeat $i/$REPEATS seed=$seed ==="

    if [[ -f "$out/summary.json" ]]; then
      echo "skip existing $out"
      continue
    fi

    if [[ "$i" -eq 1 && -n "$reuse" && -f "$reuse/summary.json" ]]; then
      echo "reuse seed41: $reuse -> $out"
      mkdir -p "$out"
      cp -f "$reuse/summary.json" "$out/summary.json"
      if [[ -f "$reuse/hintflow_one.jsonl" ]]; then
        cp -f "$reuse/hintflow_one.jsonl" "$out/hintflow_one.jsonl"
      fi
      continue
    fi

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
      --seed "$seed" \
      --workers "$workers"
  done

  echo "--- aggregate $tag ---"
  aggregate "$out_root"
done

python3 - <<'PY'
import json
from pathlib import Path

def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None

def fmt(s):
    if s is None:
        return "MISSING"
    return (
        f"HF1={100*s['em_mean']:.1f}%±{100*s['em_stdev']:.1f} "
        f"chal={100*s['challenger_em_mean']:.1f}%±{100*s.get('challenger_em_stdev',0):.1f} "
        f"base={100*s['baseline_em_mean']:.1f}%±{100*s.get('baseline_em_stdev',0):.1f} "
        f"d={100*s['paired_delta_mean']:+.1f}pp n={s['n_runs']}"
    )

root = Path("checkpoints/eval_hintflow_one_grpo_vp_step40_repeat4_20k")
print("\n=== GRPO vp step40 OOD repeat4 @20k ===")
for tag in ["dapo128", "aime24", "math500"]:
    print(f"[{tag}] {fmt(load(root / tag / 'repeat_summary.json'))}")
PY

echo "=== $(date -Is) ALL DONE ==="
echo "root: $ROOT"
