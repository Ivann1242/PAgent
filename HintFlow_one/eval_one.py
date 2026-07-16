#!/usr/bin/env python3
"""Evaluate HintFlow_one (Blind FF + conservative selector) on DAPO-Math 128."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import EVAL_PARQUET  # noqa: E402
from core import load_dapo_rows, load_jsonl, write_jsonl  # noqa: E402
from one_agent import (  # noqa: E402
    ORCH_MODEL,
    ORCH_URL,
    SOLVER_MODEL,
    HintFlowOneAgent,
)

DEFAULT_SOLVER_URLS = ["http://127.0.0.1:8006/v1"]


class _RoundRobin:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            url = self.urls[self._i % len(self.urls)]
            self._i += 1
            return url


def _eval_row(
    row: dict,
    *,
    rr: _RoundRobin,
    orch_url: str,
    orch_model: str,
    solver_model: str,
    solver_max_tokens: int,
    selector_mode: str,
    replace_threshold: float,
    seed: int,
) -> dict:
    solver_url = rr.next()
    try:
        agent = HintFlowOneAgent(
            orch_url=orch_url,
            orch_model=orch_model,
            solver_url=solver_url,
            solver_model=solver_model,
            solver_max_tokens=solver_max_tokens,
            selector_mode=selector_mode,
            replace_threshold=replace_threshold,
            solver_seed=seed + int(row["id"]) * 100,
        )
        traj = agent.run(row["problem"], gold=row["gold"])
        rec = traj.to_dict()
        rec.update(
            {
                "id": row["id"],
                "solver_url": solver_url,
                "error": None,
            }
        )
        return rec
    except Exception as e:  # noqa: BLE001
        return {
            "id": row["id"],
            "problem": row["problem"],
            "gold": row["gold"],
            "final_answer": "",
            "em": 0,
            "baseline_em": 0,
            "challenger_em": 0,
            "recovered": 0,
            "harmed": 0,
            "solver_url": solver_url,
            "error": f"{type(e).__name__}: {e}",
        }


def _run_parallel(rows: list[dict], fn, *, workers: int, desc: str) -> list[dict]:
    records: list[dict | None] = [None] * len(rows)
    idx = {row["id"]: i for i, row in enumerate(rows)}
    with ThreadPoolExecutor(max_workers=min(workers, len(rows))) as pool:
        futs = {pool.submit(fn, row): row["id"] for row in rows}
        for fut in tqdm(as_completed(futs), total=len(rows), desc=desc):
            rec = fut.result()
            records[idx[rec["id"]]] = rec
    return records  # type: ignore[return-value]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", default=None)
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--solver-urls", default=",".join(DEFAULT_SOLVER_URLS))
    p.add_argument("--orch-url", default=ORCH_URL)
    p.add_argument("--orch-model", default=ORCH_MODEL)
    p.add_argument("--solver-model", default=SOLVER_MODEL)
    p.add_argument("--solver-max-tokens", type=int, default=4096)
    p.add_argument(
        "--selector-mode",
        choices=("orch", "keep", "replace"),
        default="orch",
        help="orch=conservative selector; keep=always baseline; replace=always FF",
    )
    p.add_argument("--replace-threshold", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "checkpoints" / "eval_hintflow_one_128"),
    )
    args = p.parse_args()

    solver_urls = [u.strip() for u in args.solver_urls.split(",") if u.strip()]
    data_file = Path(args.data_file) if args.data_file else EVAL_PARQUET
    if data_file.suffix == ".parquet" or data_file == EVAL_PARQUET:
        rows = load_dapo_rows(data_file)[: args.limit]
    else:
        rows = load_jsonl(data_file)[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {data_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"HintFlow_one eval: n={len(rows)} workers={args.workers} "
        f"selector={args.selector_mode} tokens={args.solver_max_tokens} "
        f"seed={args.seed}",
        flush=True,
    )
    t0 = time.time()
    rr = _RoundRobin(solver_urls)
    records = _run_parallel(
        rows,
        lambda row: _eval_row(
            row,
            rr=rr,
            orch_url=args.orch_url,
            orch_model=args.orch_model,
            solver_model=args.solver_model,
            solver_max_tokens=args.solver_max_tokens,
            selector_mode=args.selector_mode,
            replace_threshold=args.replace_threshold,
            seed=args.seed,
        ),
        workers=args.workers,
        desc="hintflow_one",
    )
    write_jsonl(out_dir / "hintflow_one.jsonl", records)

    n = len(records)
    n_err = sum(1 for r in records if r.get("error"))
    ems = [float(r.get("em") or 0) for r in records]
    base = [float(r.get("baseline_em") or 0) for r in records]
    chal = [float(r.get("challenger_em") or 0) for r in records]
    recovered = sum(int(r.get("recovered") or 0) for r in records)
    harmed = sum(int(r.get("harmed") or 0) for r in records)
    base_correct = sum(1 for v in base if v)
    retained = sum(
        1
        for r in records
        if int(r.get("baseline_em") or 0) and int(r.get("em") or 0)
    )
    replace_n = sum(
        1
        for r in records
        if ((r.get("selection") or {}).get("decision") == "REPLACE")
        and float((r.get("selection") or {}).get("confidence") or 0)
        >= args.replace_threshold
    )

    summary = {
        "meta": {
            "data_file": str(data_file),
            "n": n,
            "workers": args.workers,
            "solver_urls": solver_urls,
            "orch_url": args.orch_url,
            "orch_model": args.orch_model,
            "solver_model": args.solver_model,
            "solver_max_tokens": args.solver_max_tokens,
            "selector_mode": args.selector_mode,
            "replace_threshold": args.replace_threshold,
            "seed": args.seed,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "hintflow_one": {
            "em": _mean(ems),
            "baseline_em": _mean(base),
            "challenger_em": _mean(chal),
            "paired_delta": _mean(ems) - _mean(base),
            "recovered": recovered,
            "harmed": harmed,
            "baseline_correct_retention": (
                retained / base_correct if base_correct else None
            ),
            "replace_count": replace_n,
            "n_error": n_err,
            "sample_error": next(
                (r.get("error") for r in records if r.get("error")), None
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
