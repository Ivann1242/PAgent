#!/usr/bin/env bash
# Draft v4 on the same holdout512 / 14B@8k / GRPO-4B protocol as HF1.
set -u
cd /home/ivaning/PAgent

OUT=checkpoints/eval_hintflow_draft_holdout512_v4
HF1=checkpoints/compare_holdout512_hf1_qwen14b_8k
RESUME=checkpoints/eval_hintflow_draft_holdout64_v4/full.jsonl
mkdir -p "${OUT}"

if [[ -f "${RESUME}" && ! -f "${OUT}/full.jsonl" ]]; then
  cp -f "${RESUME}" "${OUT}/full.jsonl"
  echo "resumed $(wc -l < "${OUT}/full.jsonl") rows from holdout64 v4"
fi

python HintFlow_draft/eval_draft.py \
  --mode full \
  --limit 512 \
  --workers 8 \
  --solver-urls http://127.0.0.1:8008/v1 \
  --solver-model qwen3-14b \
  --small-url http://127.0.0.1:8086/v1 \
  --small-model qwen3-4b-blind-ff-grpo-clean-iid-vp-step200 \
  --draft-max-tokens 8192 \
  --seed 20260905 \
  --out-dir "${OUT}"
status=$?

python - <<'PY'
import json
from pathlib import Path

def load(path, skip_error=True):
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("id") is None:
            continue
        if skip_error and rec.get("error"):
            continue
        rows[rec["id"]] = rec
    return rows

def rate(rows, ids, key):
    if not ids:
        return 0.0
    return sum(int(rows[i].get(key) or 0) for i in ids) / len(ids)

draft = load(Path("checkpoints/eval_hintflow_draft_holdout512_v4/full.jsonl"))
hf1 = load(Path("checkpoints/compare_holdout512_hf1_qwen14b_8k/hintflow_one.jsonl"))
ids = sorted(set(draft) & set(hf1))
n = max(len(ids), 1)
summary = {
    "n_paired": len(ids),
    "hf1_ref": "checkpoints/compare_holdout512_hf1_qwen14b_8k",
    "hf1_em": rate(hf1, ids, "em"),
    "hf1_baseline_em": rate(hf1, ids, "baseline_em"),
    "hf1_recovered": sum(int(hf1[i].get("recovered") or 0) for i in ids),
    "hf1_harmed": sum(int(hf1[i].get("harmed") or 0) for i in ids),
    "draft_em": rate(draft, ids, "em"),
    "draft_bare_em": rate(draft, ids, "draft_em"),
    "draft_recovered": sum(int(draft[i].get("recovered") or 0) for i in ids),
    "draft_harmed": sum(int(draft[i].get("harmed") or 0) for i in ids),
    "draft_beats_hf1": sum(
        int(bool(draft[i].get("em")) and not hf1[i].get("em")) for i in ids
    ),
    "hf1_beats_draft": sum(
        int(bool(hf1[i].get("em")) and not draft[i].get("em")) for i in ids
    ),
}
if ids:
    summary["delta_vs_hf1"] = summary["draft_em"] - summary["hf1_em"]
    summary["draft_delta_vs_bare"] = summary["draft_em"] - summary["draft_bare_em"]
    summary["hf1_delta_vs_bare"] = summary["hf1_em"] - summary["hf1_baseline_em"]
Path("checkpoints/eval_hintflow_draft_holdout512_v4/compare_hf1.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
PY
exit "${status}"
