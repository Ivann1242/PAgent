#!/usr/bin/env python3
"""Evaluate hint-gated single-OSS pipeline (bare | gated) on IID / DAPO rows."""

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
from core import append_jsonl, load_dapo_rows, load_jsonl, write_jsonl  # noqa: E402
from gate_agent import (  # noqa: E402
    MODES,
    ORCH_MODEL,
    ORCH_URL,
    SOLVER_MODEL,
    HintGateAgent,
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
    use_threshold: float,
    mode: str,
    seed: int,
) -> dict:
    solver_url = rr.next()
    try:
        agent = HintGateAgent(
            orch_url=orch_url,
            orch_model=orch_model,
            solver_url=solver_url,
            solver_model=solver_model,
            solver_max_tokens=solver_max_tokens,
            use_threshold=use_threshold,
            mode=mode,
            solver_seed=seed + int(row["id"]) * 100,
        )
        traj = agent.run(row["problem"], gold=row["gold"])
        rec = traj.to_dict()
        rec.update({"id": row["id"], "solver_url": solver_url, "error": None})
        return rec
    except Exception as e:  # noqa: BLE001
        return {
            "id": row["id"],
            "problem": row["problem"],
            "gold": row["gold"],
            "mode": mode,
            "final_answer": "",
            "em": 0,
            "used_hint": False,
            "solver_url": solver_url,
            "error": f"{type(e).__name__}: {e}",
        }


def _run_parallel(
    rows: list[dict], fn, *, workers: int, desc: str, resume_path: Path | None = None,
) -> list[dict]:
    done: dict = {}
    if resume_path is not None and resume_path.exists():
        for rec in load_jsonl(resume_path):
            if rec.get("id") is not None and not rec.get("error"):
                done[rec["id"]] = rec
        if done:
            print(f"[{desc}] resume {resume_path}: {len(done)}/{len(rows)} done", flush=True)

    todo = [row for row in rows if row["id"] not in done]
    if not todo:
        return [done[row["id"]] for row in rows]

    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
        futs = {pool.submit(fn, row): row["id"] for row in todo}
        for fut in tqdm(as_completed(futs), total=len(todo), desc=desc):
            rec = fut.result()
            done[rec["id"]] = rec
            if resume_path is not None and not rec.get("error"):
                with write_lock:
                    append_jsonl(resume_path, rec)
    return [done[row["id"]] for row in rows]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _load_rows(data_file: Path, limit: int | None) -> list[dict]:
    if data_file.suffix == ".parquet" or data_file == EVAL_PARQUET:
        rows = load_dapo_rows(data_file)
    else:
        raw = load_jsonl(data_file)
        by_id: dict = {}
        for r in raw:
            if r.get("id") not in by_id:
                by_id[r["id"]] = r
        rows = [by_id[i] for i in sorted(by_id)]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"no rows in {data_file}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--solver-urls", default=",".join(DEFAULT_SOLVER_URLS))
    p.add_argument("--orch-url", default=ORCH_URL)
    p.add_argument("--orch-model", default=ORCH_MODEL)
    p.add_argument("--solver-model", default=SOLVER_MODEL)
    p.add_argument("--solver-max-tokens", type=int, default=8192)
    p.add_argument("--use-threshold", type=float, default=0.90)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "checkpoints" / "eval_hintflow_gate"),
    )
    p.add_argument(
        "--pair-with",
        default=None,
        help="optional other-mode jsonl/dir for recover-harm vs this mode",
    )
    args = p.parse_args()

    solver_urls = [u.strip() for u in args.solver_urls.split(",") if u.strip()]
    data_file = Path(args.data_file) if args.data_file else EVAL_PARQUET
    rows = _load_rows(data_file, args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{args.mode}.jsonl"
    print(
        f"HintFlow_gate eval: n={len(rows)} workers={args.workers} mode={args.mode} "
        f"thresh={args.use_threshold} tokens={args.solver_max_tokens} seed={args.seed}",
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
            use_threshold=args.use_threshold,
            mode=args.mode,
            seed=args.seed,
        ),
        workers=args.workers,
        desc=f"gate_{args.mode}",
        resume_path=out_jsonl,
    )
    write_jsonl(out_jsonl, records)

    n = len(records)
    n_err = sum(1 for r in records if r.get("error"))
    ems = [float(r.get("em") or 0) for r in records]
    used = sum(1 for r in records if r.get("used_hint"))
    summary: dict = {
        "meta": {
            "data_file": str(data_file),
            "n": n,
            "workers": args.workers,
            "solver_urls": solver_urls,
            "orch_url": args.orch_url,
            "orch_model": args.orch_model,
            "solver_model": args.solver_model,
            "solver_max_tokens": args.solver_max_tokens,
            "use_threshold": args.use_threshold,
            "mode": args.mode,
            "seed": args.seed,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        args.mode: {
            "em": _mean(ems),
            "used_hint_rate": used / n if n else 0.0,
            "used_hint_count": used,
            "n_error": n_err,
            "sample_error": next(
                (r.get("error") for r in records if r.get("error")), None
            ),
        },
    }

    # Paired recover/harm vs bare (or --pair-with)
    pair_path = None
    if args.pair_with:
        pair_path = Path(args.pair_with)
        if pair_path.is_dir():
            # prefer bare.jsonl in that dir
            cand = pair_path / "bare.jsonl"
            pair_path = cand if cand.exists() else pair_path
    elif args.mode == "gated":
        cand = out_dir / "bare.jsonl"
        if cand.exists():
            pair_path = cand

    if pair_path is not None and pair_path.exists():
        other = {r["id"]: r for r in load_jsonl(pair_path) if r.get("id") is not None}
        cur = {r["id"]: r for r in records if r.get("id") is not None}
        ids = sorted(set(other) & set(cur))
        rec = harm = 0
        for i in ids:
            b = int(bool(other[i].get("em")))
            g = int(bool(cur[i].get("em")))
            if g and not b:
                rec += 1
            if b and not g:
                harm += 1
        base_em = _mean([float(other[i].get("em") or 0) for i in ids])
        gate_em = _mean([float(cur[i].get("em") or 0) for i in ids])
        summary["paired_vs_bare"] = {
            "n": len(ids),
            "bare_em": base_em,
            "gated_em": gate_em,
            "paired_delta": gate_em - base_em,
            "recovered": rec,
            "harmed": harm,
            "pair_file": str(pair_path),
        }

    summary_path = out_dir / f"summary_{args.mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    # also merge into summary.json if both modes exist
    merged_path = out_dir / "summary.json"
    merged = {}
    if merged_path.exists():
        try:
            merged = json.loads(merged_path.read_text())
        except json.JSONDecodeError:
            merged = {}
    merged.setdefault("modes", {})
    merged["modes"][args.mode] = summary[args.mode]
    merged["meta_last"] = summary["meta"]
    if "paired_vs_bare" in summary:
        merged["paired_vs_bare"] = summary["paired_vs_bare"]
    merged_path.write_text(json.dumps(merged, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
