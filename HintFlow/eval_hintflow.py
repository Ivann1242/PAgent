#!/usr/bin/env python3
"""Evaluate HintFlow on 128 problems with 4x OSS (batch 8 each → 32 workers)."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import EVAL_PARQUET  # noqa: E402
from core import load_dapo_rows, load_jsonl, rollout, write_jsonl  # noqa: E402
from HintFlowAgent import (  # noqa: E402
    ORCH_MODEL,
    ORCH_URL,
    SOLVER_MODEL,
    HintFlowAgent,
)

DEFAULT_SOLVER_URLS = [
    "http://127.0.0.1:8006/v1",
    "http://127.0.0.1:8007/v1",
    "http://127.0.0.1:8008/v1",
    "http://127.0.0.1:8009/v1",
]


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


def _compact_traj(row: dict, traj) -> dict:
    steps = []
    for s in traj.steps:
        r = s.review
        steps.append({
            "index": s.index,
            "instruction": s.instruction,
            "observation": s.observation,
            "injected_prompt": s.injected_prompt,
            "retried": s.retried,
            "is_final": s.is_final,
            "review": None if r is None else {
                "summary": r.summary,
                "status": r.status,
                "issue": r.issue,
                "hint": r.hint,
                "action": r.action,
            },
        })
    return {
        "id": row["id"],
        "problem": row["problem"],
        "gold": row["gold"],
        "final_answer": traj.final_answer,
        "em": int(traj.em or 0),
        "n_steps": len(traj.steps),
        "running_summary": traj.running_summary,
        "plan_nodes": [
            {
                "instruction": n.instruction,
                "inject_after": n.inject_after,
                "is_final": n.is_final,
            }
            for n in (traj.plan.nodes if traj.plan else [])
        ],
        "steps": steps,
        "error": None,
    }


def _eval_hintflow_row(
    row: dict,
    *,
    rr: _RoundRobin,
    orch_url: str,
    orch_model: str,
    solver_model: str,
) -> dict:
    solver_url = rr.next()
    try:
        agent = HintFlowAgent(
            orch_url=orch_url,
            orch_model=orch_model,
            solver_url=solver_url,
            solver_model=solver_model,
        )
        traj = agent.run(row["problem"], gold=row["gold"])
        rec = _compact_traj(row, traj)
        rec["solver_url"] = solver_url
        return rec
    except Exception as e:
        return {
            "id": row["id"],
            "problem": row["problem"],
            "gold": row["gold"],
            "final_answer": "",
            "em": 0,
            "n_steps": 0,
            "running_summary": "",
            "plan_nodes": [],
            "steps": [],
            "solver_url": solver_url,
            "error": f"{type(e).__name__}: {e}",
        }


def _eval_baseline_row(row: dict, *, rr: _RoundRobin, solver_model: str) -> dict:
    url = rr.next()
    client = OpenAI(base_url=url, api_key="EMPTY")
    try:
        r = rollout(
            client, solver_model, row["problem"], row["gold"], "baseline",
            max_tokens=8192, protocol="native",
        )
        return {
            "id": row["id"],
            "problem": row["problem"],
            "gold": row["gold"],
            "pred": r.get("pred", ""),
            "em": int(r.get("em", 0)),
            "solver_url": url,
            "error": None,
        }
    except Exception as e:
        return {
            "id": row["id"],
            "problem": row["problem"],
            "gold": row["gold"],
            "pred": "",
            "em": 0,
            "solver_url": url,
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


def _em(recs: list[dict]) -> float:
    if not recs:
        return 0.0
    return sum(int(r.get("em") or 0) for r in recs) / len(recs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-file",
        default=None,
        help="default: EVAL_PARQUET (same 128-set as prior blind FF evals)",
    )
    p.add_argument("--limit", type=int, default=128)
    p.add_argument(
        "--workers", type=int, default=32,
        help="parallel problems (default 32 = 4 OSS GPUs × batch 8)",
    )
    p.add_argument(
        "--solver-urls",
        default=",".join(DEFAULT_SOLVER_URLS),
        help="comma-separated OSS OpenAI base URLs",
    )
    p.add_argument("--orch-url", default=ORCH_URL)
    p.add_argument("--orch-model", default=ORCH_MODEL)
    p.add_argument("--solver-model", default=SOLVER_MODEL)
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "checkpoints" / "eval_hintflow_128"),
    )
    p.add_argument("--skip-baseline", action="store_true")
    args = p.parse_args()

    solver_urls = [u.strip() for u in args.solver_urls.split(",") if u.strip()]
    if not solver_urls:
        raise SystemExit("no solver urls")

    data_file = Path(args.data_file) if args.data_file else EVAL_PARQUET
    if data_file.suffix == ".parquet" or data_file == EVAL_PARQUET:
        from core import load_dapo_rows
        rows = load_dapo_rows(data_file)[: args.limit]
    else:
        rows = load_jsonl(data_file)[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {data_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"HintFlow eval: n={len(rows)} workers={args.workers} "
        f"solvers={len(solver_urls)} orch={args.orch_url} data={data_file}",
        flush=True,
    )
    t0 = time.time()

    summary: dict = {
        "meta": {
            "data_file": str(data_file),
            "n": len(rows),
            "workers": args.workers,
            "solver_urls": solver_urls,
            "orch_url": args.orch_url,
            "orch_model": args.orch_model,
            "solver_model": args.solver_model,
            "batch_per_gpu_target": 8,
            "n_gpus": len(solver_urls),
        }
    }

    if not args.skip_baseline:
        rr_b = _RoundRobin(solver_urls)
        baseline = _run_parallel(
            rows,
            lambda row: _eval_baseline_row(
                row, rr=rr_b, solver_model=args.solver_model,
            ),
            workers=args.workers,
            desc="live_baseline",
        )
        write_jsonl(out_dir / "live_baseline.jsonl", baseline)
        n_err_b = sum(1 for r in baseline if r.get("error"))
        sample_err_b = next((r.get("error") for r in baseline if r.get("error")), None)
        summary["live_baseline"] = {
            "em": _em(baseline),
            "n": len(baseline),
            "n_error": n_err_b,
            "sample_error": sample_err_b,
        }
        print(f"baseline EM={summary['live_baseline']['em']*100:.2f}%", flush=True)
        if sample_err_b:
            print(f"baseline sample_error: {sample_err_b}", flush=True)

    rr_h = _RoundRobin(solver_urls)
    hintflow = _run_parallel(
        rows,
        lambda row: _eval_hintflow_row(
            row,
            rr=rr_h,
            orch_url=args.orch_url,
            orch_model=args.orch_model,
            solver_model=args.solver_model,
        ),
        workers=args.workers,
        desc="hintflow",
    )
    write_jsonl(out_dir / "hintflow.jsonl", hintflow)

    actions: dict[str, int] = {}
    n_steps = []
    for r in hintflow:
        n_steps.append(r.get("n_steps") or 0)
        for s in r.get("steps") or []:
            a = ((s.get("review") or {}).get("action")) or "NONE"
            actions[a] = actions.get(a, 0) + 1

    n_err_h = sum(1 for r in hintflow if r.get("error"))
    sample_err_h = next((r.get("error") for r in hintflow if r.get("error")), None)
    summary["hintflow"] = {
        "em": _em(hintflow),
        "n": len(hintflow),
        "n_error": n_err_h,
        "sample_error": sample_err_h,
        "avg_steps": (sum(n_steps) / len(n_steps)) if n_steps else 0.0,
        "action_counts": actions,
    }
    if sample_err_h:
        print(f"hintflow sample_error: {sample_err_h}", flush=True)
    if "live_baseline" in summary:
        summary["hintflow_vs_live_baseline"] = (
            summary["hintflow"]["em"] - summary["live_baseline"]["em"]
        )

    # paired flips vs baseline
    if not args.skip_baseline:
        base_by_id = {r["id"]: r for r in baseline}
        hf_only = base_only = 0
        for r in hintflow:
            b = base_by_id.get(r["id"])
            if not b:
                continue
            be, he = int(b.get("em") or 0), int(r.get("em") or 0)
            if he and not be:
                hf_only += 1
            elif be and not he:
                base_only += 1
        summary["paired"] = {
            "hintflow_only_correct": hf_only,
            "baseline_only_correct": base_only,
        }

    summary["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
