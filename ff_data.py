"""Build free-form SFT labels from action-router labels."""

from __future__ import annotations

import json
from pathlib import Path

from core import ACTION_SPACE, load_jsonl, write_jsonl


def build_ff_labels(
    labels_file: Path,
    out_file: Path,
) -> dict:
    rows = load_jsonl(labels_file)
    out = []
    for row in rows:
        action = row["label_action"]
        hint = ACTION_SPACE.get(action, "")
        out.append({
            "id": row["id"],
            "problem": row["problem"],
            "gold": row["gold"],
            "label_action": action,
            "label_hint": hint,
        })
    write_jsonl(out_file, out)
    stats = {
        "n": len(out),
        "labels_file": str(labels_file),
        "out_file": str(out_file),
        "empty_hint": sum(1 for r in out if not r["label_hint"].strip()),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    stats_path = out_file.parent / "ff_labels_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"ff labels: {len(out)} -> {out_file}")
    print(f"empty hints (baseline): {stats['empty_hint']}")
    return stats
