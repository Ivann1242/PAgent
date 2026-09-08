#!/usr/bin/env bash
set -u
cd /home/ivaning/PAgent
OUT=checkpoints/eval_hintflow_draft_holdout64_v2
mkdir -p "${OUT}"
python HintFlow_draft/eval_draft.py \
  --mode full \
  --limit 64 \
  --workers 8 \
  --solver-urls http://127.0.0.1:8008/v1 \
  --planner-max-tokens 512 \
  --out-dir "${OUT}"
status=$?
python - <<'PY'
import json
from pathlib import Path

def load(path):
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("id") is not None and not rec.get("error"):
            rows[rec["id"]] = rec
    return rows

old = load(Path("checkpoints/eval_hintflow_draft_holdout64_v1/full.jsonl"))
new = load(Path("checkpoints/eval_hintflow_draft_holdout64_v2/full.jsonl"))
ids = sorted(set(old) & set(new))
n = max(len(ids), 1)
summary = {
    "n_paired": len(ids),
    "v1_em": sum(int(old[i].get("em") or 0) for i in ids) / n if ids else 0,
    "v2_em": sum(int(new[i].get("em") or 0) for i in ids) / n if ids else 0,
    "v2_draft_em": sum(int(new[i].get("draft_em") or 0) for i in ids) / n if ids else 0,
    "v2_reconstructed_em": sum(int(new[i].get("reconstructed_em") or 0) for i in ids) / n if ids else 0,
    "v1_recovered": sum(int(old[i].get("recovered") or 0) for i in ids),
    "v1_harmed": sum(int(old[i].get("harmed") or 0) for i in ids),
    "v2_recovered": sum(int(new[i].get("recovered") or 0) for i in ids),
    "v2_harmed": sum(int(new[i].get("harmed") or 0) for i in ids),
    "v2_keep_draft": sum(int(new[i].get("exit_source") == "DRAFT") for i in ids),
    "keep": False,
}
if ids:
    summary["delta_vs_v1"] = summary["v2_em"] - summary["v1_em"]
    summary["keep"] = (
        summary["v2_em"] > summary["v1_em"]
        and summary["v2_harmed"] <= summary["v1_harmed"]
    ) or (
        summary["v2_em"] >= summary["v1_em"]
        and summary["v2_harmed"] < summary["v1_harmed"]
    )
Path("checkpoints/eval_hintflow_draft_holdout64_v2/compare_v1.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
PY
exit "${status}"
