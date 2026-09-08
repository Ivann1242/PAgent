#!/usr/bin/env python3
"""Evaluate the five-round interact protocol."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import append_jsonl, load_jsonl, write_jsonl
from HintFlow_interact.agent import InteractAgent
from HintFlow_interact.prompts import ACTS


class _RoundRobin:
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            url = self.urls[self._i % len(self.urls)]
            self._i += 1
            return url


def _eval_row(row: dict, *, rr: _RoundRobin, args) -> dict:
    solver_url = rr.next()
    try:
        agent = InteractAgent(
            small_url=args.small_url,
            small_model=args.small_model,
            solver_url=solver_url,
            solver_model=args.solver_model,
            solver_max_tokens=args.solver_max_tokens,
            replace_threshold=args.replace_threshold,
            seed=args.seed + int(row["id"]) * 100,
        )
        rec = agent.run(row["problem"], gold=row.get("gold") or "").to_dict()
        rec.update({"id": row["id"], "solver_url": solver_url, "error": None})
        return rec
    except Exception as exc:  # noqa: BLE001
        return {
            "id": row["id"],
            "problem": row.get("problem", ""),
            "gold": row.get("gold", ""),
            "turns": [],
            "final_answer": "",
            "baseline_em": 0,
            "em": 0,
            "recovered": 0,
            "harmed": 0,
            "n_parseable": 0,
            "replace_count": 0,
            "solver_url": solver_url,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", default="data/dapo_holdout_512.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--solver-urls", default="http://127.0.0.1:8008/v1")
    p.add_argument("--small-url", default="http://127.0.0.1:8086/v1")
    p.add_argument("--small-model", default="qwen3-4b-blind-ff-grpo-clean-iid-vp-step200")
    p.add_argument("--solver-model", default="qwen3-14b")
    p.add_argument("--solver-max-tokens", type=int, default=8192)
    p.add_argument("--replace-threshold", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--out-dir", default="checkpoints/eval_hintflow_interact_holdout512")
    args = p.parse_args()

    rows = load_jsonl(Path(args.data_file))
    by_id: dict = {}
    for r in rows:
        if r.get("id") not in by_id:
            by_id[r["id"]] = r
    rows = [by_id[i] for i in sorted(by_id)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {args.data_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "interact.jsonl"
    done: dict = {}
    if out_jsonl.exists():
        for rec in load_jsonl(out_jsonl):
            if rec.get("id") is not None and not rec.get("error"):
                done[rec["id"]] = rec
    todo = [r for r in rows if r["id"] not in done]
    urls = [u.strip() for u in args.solver_urls.split(",") if u.strip()]
    rr = _RoundRobin(urls)
    print(
        f"interact eval n={len(rows)} todo={len(todo)} workers={args.workers} "
        f"small={args.small_model} solver={args.solver_model}",
        flush=True,
    )
    t0 = time.time()
    lock = threading.Lock()
    if todo:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(todo))) as pool:
            futs = {pool.submit(_eval_row, row, rr=rr, args=args): row["id"] for row in todo}
            for fut in tqdm(as_completed(futs), total=len(todo), desc="interact"):
                rec = fut.result()
                done[rec["id"]] = rec
                if not rec.get("error"):
                    with lock:
                        append_jsonl(out_jsonl, rec)
    records = [done[r["id"]] for r in rows if r["id"] in done]
    write_jsonl(out_jsonl, records)

    n = len(records)
    acts = Counter()
    became = Counter()
    for rec in records:
        for t in rec.get("turns") or []:
            acts[t.get("act")] += 1
            if t.get("became_incumbent"):
                became[t.get("act")] += 1
    base = [float(r.get("baseline_em") or 0) for r in records]
    ems = [float(r.get("em") or 0) for r in records]
    summary = {
        "meta": {
            "data_file": args.data_file,
            "n": n,
            "requested_n": len(rows),
            "workers": args.workers,
            "solver_urls": urls,
            "small_url": args.small_url,
            "small_model": args.small_model,
            "solver_model": args.solver_model,
            "solver_max_tokens": args.solver_max_tokens,
            "replace_threshold": args.replace_threshold,
            "seed": args.seed,
            "elapsed_sec": round(time.time() - t0, 1),
            "acts": ACTS,
        },
        "interact": {
            "em": sum(ems) / n if n else 0,
            "baseline_em": sum(base) / n if n else 0,
            "paired_delta": (sum(ems) - sum(base)) / n if n else 0,
            "recovered": sum(int(r.get("recovered") or 0) for r in records),
            "harmed": sum(int(r.get("harmed") or 0) for r in records),
            "replace_count": sum(int(r.get("replace_count") or 0) for r in records),
            "mean_parseable_solves": (
                sum(int(r.get("n_parseable") or 0) for r in records) / n if n else 0
            ),
            "became_incumbent": dict(became),
            "n_error": sum(1 for r in records if r.get("error")),
            "sample_error": next((r.get("error") for r in records if r.get("error")), None),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
