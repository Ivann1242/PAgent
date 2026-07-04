"""Build offline DPO preference pairs from exhaustive action rollouts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from core import load_jsonl, write_jsonl


def _completion_for_action(action: str) -> str:
    return json.dumps({"action": action}, ensure_ascii=False)


def build_dpo_pairs(
    rollouts_file: Path,
    out_file: Path,
    *,
    category: str = "signal",
) -> dict:
    """All (correct x incorrect) action pairs for questions with mixed EM."""
    records = load_jsonl(rollouts_file)
    pairs: list[dict] = []
    per_q = 0

    for row in records:
        if row.get("category") != category:
            continue
        wins = [a for a, em in row["action_ems"].items() if em == 1]
        loses = [a for a, em in row["action_ems"].items() if em == 0]
        if not wins or not loses:
            continue
        prompt = row.get("router_prompt") or row["problem"]
        n = 0
        for w in wins:
            for l in loses:
                pairs.append({
                    "id": row["id"],
                    "prompt": prompt,
                    "chosen": _completion_for_action(w),
                    "rejected": _completion_for_action(l),
                    "win_action": w,
                    "lose_action": l,
                })
                n += 1
        per_q += n

    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_file, pairs)

    win_counter = Counter(p["win_action"] for p in pairs)
    lose_counter = Counter(p["lose_action"] for p in pairs)
    stats = {
        "rollouts_file": str(rollouts_file),
        "out_file": str(out_file),
        "category": category,
        "n_questions": sum(1 for r in records if r.get("category") == category),
        "n_pairs": len(pairs),
        "avg_pairs_per_question": len(pairs) / max(
            sum(1 for r in records if r.get("category") == category), 1
        ),
        "win_action_distribution": dict(win_counter),
        "lose_action_distribution": dict(lose_counter),
    }
    (out_file.parent / "dpo_pairs_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"questions={stats['n_questions']} pairs={stats['n_pairs']}")
    print(f"win actions: {stats['win_action_distribution']}")
    print(f"-> {out_file}")
    return stats
