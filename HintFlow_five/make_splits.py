#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from core import load_jsonl, write_jsonl


def fingerprint(row: dict) -> str:
    return hashlib.sha256(row["problem"].strip().encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/train.jsonl")
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--iid-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--out-dir", default="checkpoints/hintflow_five_data")
    args = p.parse_args()
    rows = load_jsonl(Path(args.source))
    unique = {}
    for row in rows:
        unique.setdefault(fingerprint(row), row)
    rows = list(unique.values())
    random.Random(args.seed).shuffle(rows)
    need = args.train_size + args.iid_size
    if len(rows) < need:
        raise SystemExit(f"need {need} unique questions, found {len(rows)}")
    train, iid = rows[: args.train_size], rows[args.train_size:need]
    out = Path(args.out_dir)
    write_jsonl(out / "teacher_questions.jsonl", train)
    write_jsonl(out / "iid_512.jsonl", iid)
    meta = {
        "source": args.source,
        "seed": args.seed,
        "n_source": len(rows),
        "n_train_questions": len(train),
        "n_iid_questions": len(iid),
        "overlap": len({fingerprint(x) for x in train} & {fingerprint(x) for x in iid}),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
