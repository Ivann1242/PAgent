#!/usr/bin/env python3
"""Build a leakage-free DAPO holdout split for larger evals.

Uses the same shuffle as `run.py prepare` (seed=42): train = [0, train_n),
val = [train_n, train_n+val_n). Holdout is sampled from the remainder, then
filtered by normalized problem text against train/val/DAPO-128 eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import datasets

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EVAL_PARQUET  # noqa: E402
from core import extract_raw_question, load_dapo_rows, load_jsonl, write_jsonl  # noqa: E402

DATA_SOURCE = "BytedTsinghua-SIA/DAPO-Math-17k"


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _gold_of(ex) -> str:
    gold = ex["reward_model"]["ground_truth"]
    if hasattr(gold, "__len__") and not isinstance(gold, str):
        gold = gold[0]
    return str(gold)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", default=str(_ROOT / "data" / "train.jsonl"))
    p.add_argument("--val-file", default=str(_ROOT / "data" / "val.jsonl"))
    p.add_argument("--eval-parquet", default=str(EVAL_PARQUET))
    p.add_argument("--out-file", default=str(_ROOT / "data" / "dapo_holdout_512.jsonl"))
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--prepare-seed", type=int, default=42)
    p.add_argument("--train-size", type=int, default=17000)
    p.add_argument("--val-size", type=int, default=256)
    args = p.parse_args()

    train = load_jsonl(Path(args.train_file))
    val = load_jsonl(Path(args.val_file))
    dapo128 = load_dapo_rows(Path(args.eval_parquet))

    blocked = {_norm(r["problem"]) for r in train}
    blocked |= {_norm(r["problem"]) for r in val}
    blocked |= {_norm(r["problem"]) for r in dapo128}

    # Sanity: prepare shuffle must match local train[0].
    full = datasets.load_dataset(DATA_SOURCE, split="train").shuffle(seed=args.prepare_seed)
    q0 = extract_raw_question(full[0]["prompt"][0]["content"])
    if _norm(q0) != _norm(train[0]["problem"]):
        raise SystemExit(
            "prepare shuffle mismatch with data/train.jsonl[0]; refuse to build holdout"
        )

    skip = args.train_size + args.val_size
    rows: list[dict] = []
    scanned = 0
    skipped_blocked = 0
    for i in range(skip, len(full)):
        scanned += 1
        ex = full[i]
        problem = extract_raw_question(ex["prompt"][0]["content"])
        key = _norm(problem)
        if key in blocked:
            skipped_blocked += 1
            continue
        rows.append(
            {
                "id": int(i),
                "data_source": DATA_SOURCE,
                "problem": problem,
                "gold": _gold_of(ex),
                "split": "holdout",
                "prepare_seed": args.prepare_seed,
                "shuffled_index": int(i),
            }
        )
        blocked.add(key)  # de-dup within holdout stream
        if len(rows) >= args.n:
            break

    if len(rows) < args.n:
        raise SystemExit(f"only collected {len(rows)}/{args.n} holdout rows")

    out = Path(args.out_file)
    write_jsonl(out, rows)
    stats = {
        "out_file": str(out),
        "n": len(rows),
        "prepare_seed": args.prepare_seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "skip_prefix": skip,
        "scanned_after_prefix": scanned,
        "skipped_blocked_or_dup": skipped_blocked,
        "blocked_from_train": len(train),
        "blocked_from_val": len(val),
        "blocked_from_dapo128": len(dapo128),
        "overlap_recheck_train": sum(
            1 for r in rows if _norm(r["problem"]) in {_norm(x["problem"]) for x in train}
        ),
        "overlap_recheck_val": sum(
            1 for r in rows if _norm(r["problem"]) in {_norm(x["problem"]) for x in val}
        ),
        "overlap_recheck_dapo128": sum(
            1 for r in rows if _norm(r["problem"]) in {_norm(x["problem"]) for x in dapo128}
        ),
        "id_range": [rows[0]["id"], rows[-1]["id"]],
    }
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
