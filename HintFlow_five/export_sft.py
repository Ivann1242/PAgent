#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core import load_jsonl, write_jsonl
from HintFlow_five.common import N_TURNS


def fp(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="checkpoints/hintflow_five_data/teacher_trajectories.jsonl")
    p.add_argument("--iid", default="checkpoints/hintflow_five_data/iid_512.jsonl")
    p.add_argument("--out", default="checkpoints/hintflow_five_data/sft_turns.jsonl")
    args = p.parse_args()
    teachers = load_jsonl(Path(args.teacher))
    iid = load_jsonl(Path(args.iid))
    iid_fp = {fp(r["problem"]) for r in iid}
    out = []
    question_ids = set()
    for row in teachers:
        if fp(row["problem"]) in iid_fp:
            raise SystemExit(f"train/IID leakage for id={row['id']}")
        turns = row.get("turns", [])
        if len(turns) != N_TURNS:
            raise SystemExit(f"id={row['id']} has {len(turns)} turns")
        accepted = []
        question_ids.add(row["id"])
        for turn in turns:
            out.append({
                "id": row["id"],
                "problem": row["problem"],
                "gold": row["gold"],
                "accepted_steps": list(accepted),
                "turn": int(turn["turn"]),
                "remaining_turns": N_TURNS - int(turn["turn"]) + 1,
                "label_hint": turn["hint"],
            })
            accepted.append(turn["step"])
    write_jsonl(Path(args.out), out)
    meta = {
        "n_questions": len(question_ids),
        "n_turn_samples": len(out),
        "expected_turn_samples": len(question_ids) * N_TURNS,
        "n_iid": len(iid),
        "train_iid_overlap": 0,
    }
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
