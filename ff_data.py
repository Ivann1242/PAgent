"""Build free-form SFT labels from action-router labels."""

from __future__ import annotations

import json
import random
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


def build_blind_mixed_labels(
    baselines_file: Path,
    oracle_labels_file: Path,
    out_file: Path,
    *,
    empty_ratio: float = 0.5,
    seed: int = 42,
) -> dict:
    """Flip hints + subsampled empty hint for baseline-correct questions."""
    baselines = load_jsonl(baselines_file)
    flip_rows = load_jsonl(oracle_labels_file)

    correct_rows = [
        {
            "id": row["id"],
            "problem": row["problem"],
            "gold": row.get("gold") or row.get("gold_answer", ""),
            "label_hint": "",
            "source": "baseline_correct",
        }
        for row in baselines
        if row.get("em") == 1
    ]

    n_flip = len(flip_rows)
    if not (0.0 < empty_ratio < 1.0):
        raise ValueError(f"empty_ratio must be in (0, 1), got {empty_ratio}")
    n_empty = min(
        len(correct_rows),
        round(n_flip * empty_ratio / (1.0 - empty_ratio)),
    )

    rng = random.Random(seed)
    empty_pool = list(correct_rows)
    rng.shuffle(empty_pool)
    empty_rows = empty_pool[:n_empty]

    out = empty_rows + [{**row, "source": "blind_flip"} for row in flip_rows]
    rng.shuffle(out)

    write_jsonl(out_file, out)
    stats = {
        "n_total": len(out),
        "n_empty": len(empty_rows),
        "n_flip": n_flip,
        "empty_ratio_target": empty_ratio,
        "empty_ratio_actual": len(empty_rows) / len(out) if out else 0.0,
        "n_baseline_correct_pool": len(correct_rows),
        "seed": seed,
        "baselines_file": str(baselines_file),
        "oracle_labels_file": str(oracle_labels_file),
        "out_file": str(out_file),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    stats_path = out_file.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(
        f"mixed blind labels: {len(out)} "
        f"(empty={stats['n_empty']} flip={stats['n_flip']} "
        f"ratio={stats['empty_ratio_actual']:.1%}) -> {out_file}"
    )
    return stats
