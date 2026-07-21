#!/usr/bin/env python3
"""Create immutable problem-hash train/dev/final manifests for residual feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EVAL_PARQUET  # noqa: E402
from core import load_dapo_rows, load_jsonl, write_jsonl  # noqa: E402


def problem_hash(problem: str) -> str:
    normalized = " ".join((problem or "").split()).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe(rows: list[dict], excluded: set[str]) -> list[dict]:
    seen = set(excluded)
    out = []
    for row in rows:
        digest = problem_hash(str(row.get("problem") or ""))
        if digest in seen:
            continue
        seen.add(digest)
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", default=str(_ROOT / "data" / "train.jsonl"))
    p.add_argument("--val-file", default=str(_ROOT / "data" / "val.jsonl"))
    p.add_argument("--final-file", default=str(EVAL_PARQUET))
    p.add_argument("--dev-size", type=int, default=512)
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "checkpoints" / "residual_splits"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.json"
    source_paths = {
        "train": Path(args.train_file).resolve(),
        "val": Path(args.val_file).resolve(),
        "final": Path(args.final_file).resolve(),
    }
    source_hashes = {name: file_hash(path) for name, path in source_paths.items()}
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_hashes = existing.get("source_hashes") or {}
        existing_dev_n = ((existing.get("splits") or {}).get("dev") or {}).get("n")
        if existing_hashes == source_hashes and existing_dev_n == args.dev_size:
            print(json.dumps({"status": "unchanged", "manifest": str(manifest_path)}))
            return
        raise SystemExit("split manifest already exists with different inputs; use --force")

    train = load_jsonl(source_paths["train"])
    val = load_jsonl(source_paths["val"])
    final_path = Path(args.final_file)
    final = (
        load_dapo_rows(final_path)
        if final_path.suffix == ".parquet"
        else load_jsonl(final_path)
    )
    final_hashes = {problem_hash(str(row.get("problem") or "")) for row in final}
    val = _dedupe(val, final_hashes)
    train = _dedupe(train, final_hashes | {problem_hash(r["problem"]) for r in val})

    ranked_train = sorted(
        train,
        key=lambda row: hashlib.sha1(
            f"{args.seed}:{problem_hash(row['problem'])}".encode("utf-8")
        ).hexdigest(),
    )
    take_train = max(args.dev_size - len(val), 0)
    dev = (val + ranked_train[:take_train])[: args.dev_size]
    dev_hashes = {problem_hash(row["problem"]) for row in dev}
    train_rows = [
        row for row in train if problem_hash(row["problem"]) not in dev_hashes
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "final.jsonl", final)
    manifest = {
        "version": 1,
        "seed": args.seed,
        "sources": {
            "train": str(source_paths["train"]),
            "val": str(source_paths["val"]),
            "final": str(final_path.resolve()),
        },
        "source_hashes": source_hashes,
        "splits": {
            "train": {
                "n": len(train_rows),
                "file": str((out_dir / "train.jsonl").resolve()),
                "problem_hashes": sorted(
                    problem_hash(row["problem"]) for row in train_rows
                ),
            },
            "dev": {
                "n": len(dev),
                "problem_hashes": sorted(dev_hashes),
                "file": str((out_dir / "dev.jsonl").resolve()),
            },
            "final": {
                "n": len(final),
                "problem_hashes": sorted(final_hashes),
                "file": str((out_dir / "final.jsonl").resolve()),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train": len(train_rows),
                "dev": len(dev),
                "final": len(final),
                "manifest": str(out_dir / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
