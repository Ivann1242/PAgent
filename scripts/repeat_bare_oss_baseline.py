#!/usr/bin/env python3
"""Repeat bare OSS baseline on the DAPO-Math 128 set (no orch / no hint)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import (  # noqa: E402
    build_large_prompt,
    exact_match,
    extract_final_answer,
    load_dapo_rows,
    load_jsonl,
    write_jsonl,
)


def _message_text(resp) -> str:
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    if content:
        return str(content).strip()
    for name in ("reasoning", "reasoning_content"):
        value = getattr(msg, name, None)
        if value:
            return str(value).strip()
    return ""


def _eval_one(
    row: dict,
    *,
    client: OpenAI,
    model: str,
    max_tokens: int,
    seed: int | None,
) -> dict:
    prompt = build_large_prompt(row["problem"], "")
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["extra_body"] = {"seed": int(seed)}
    try:
        resp = client.chat.completions.create(**kwargs)
        text = _message_text(resp)
        content = getattr(resp.choices[0].message, "content", None) or ""
        reasoning = (
            getattr(resp.choices[0].message, "reasoning", None)
            or getattr(resp.choices[0].message, "reasoning_content", None)
            or ""
        )
        pred = extract_final_answer(text)
        em = exact_match(pred, row["gold"])
        return {
            "id": row["id"],
            "problem": row["problem"],
            "gold": row["gold"],
            "prompt": prompt,
            "pred": pred,
            "em": int(em),
            "content_len": len(str(content)),
            "reasoning_len": len(str(reasoning)),
            "finish_reason": resp.choices[0].finish_reason,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "id": row["id"],
            "problem": row["problem"],
            "gold": row["gold"],
            "prompt": prompt,
            "pred": "",
            "em": 0,
            "content_len": 0,
            "reasoning_len": 0,
            "finish_reason": None,
            "error": f"{type(e).__name__}: {e}",
        }


def run_once(
    rows: list[dict],
    *,
    solver_url: str,
    model: str,
    workers: int,
    max_tokens: int,
    seed: int | None,
    desc: str,
) -> list[dict]:
    client = OpenAI(base_url=solver_url, api_key="EMPTY")
    lock = threading.Lock()
    records: list[dict | None] = [None] * len(rows)
    idx = {row["id"]: i for i, row in enumerate(rows)}

    def _job(row: dict) -> dict:
        # Per-problem seed keeps temp=0 runs auditable without forcing bit-exact decode.
        local_seed = None if seed is None else seed + int(row["id"])
        return _eval_one(
            row,
            client=client,
            model=model,
            max_tokens=max_tokens,
            seed=local_seed,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_job, row): row["id"] for row in rows}
        with tqdm(total=len(rows), desc=desc) as bar:
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    records[idx[rec["id"]]] = rec
                bar.update(1)
    return records  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", default="checkpoints/residual_splits/final.jsonl")
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--solver-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--solver-model", default="gpt-oss-20b")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default="checkpoints/baseline_oss_repeat_128",
    )
    args = p.parse_args()

    data_file = Path(args.data_file)
    if data_file.suffix == ".parquet":
        rows = load_dapo_rows(data_file)[: args.limit]
    else:
        rows = load_jsonl(data_file)[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {data_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summaries = []
    t_all = time.time()
    for run_i in range(1, args.repeats + 1):
        seed = args.base_seed + 1000 * (run_i - 1)
        run_dir = out_dir / f"run{run_i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n=== bare OSS baseline run{run_i}/{args.repeats} "
            f"n={len(rows)} seed={seed} max_tokens={args.max_tokens} ===",
            flush=True,
        )
        t0 = time.time()
        recs = run_once(
            rows,
            solver_url=args.solver_url,
            model=args.solver_model,
            workers=args.workers,
            max_tokens=args.max_tokens,
            seed=seed,
            desc=f"baseline_run{run_i}",
        )
        write_jsonl(run_dir / "live_baseline.jsonl", recs)
        n_err = sum(1 for r in recs if r.get("error"))
        em = sum(int(r["em"]) for r in recs) / len(recs)
        summary = {
            "run": run_i,
            "seed": seed,
            "n": len(recs),
            "em": em,
            "n_correct": sum(int(r["em"]) for r in recs),
            "n_error": n_err,
            "sample_error": next((r.get("error") for r in recs if r.get("error")), None),
            "only_fa_content": sum(
                1
                for r in recs
                if (r.get("content_len") or 0) > 0
                and (r.get("content_len") or 0) < 40
                and (r.get("pred") or "").strip() != ""
            ),
            "finish_length": sum(1 for r in recs if r.get("finish_reason") == "length"),
            "elapsed_sec": round(time.time() - t0, 1),
            "meta": {
                "data_file": str(data_file),
                "solver_url": args.solver_url,
                "solver_model": args.solver_model,
                "max_tokens": args.max_tokens,
                "workers": args.workers,
                "hint": "",
                "protocol": "bare_oss_native",
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        run_summaries.append(summary)
        print(
            f"run{run_i}: EM={em*100:.2f}% ({summary['n_correct']}/{summary['n']}) "
            f"err={n_err} elapsed={summary['elapsed_sec']}s",
            flush=True,
        )

    ems = [s["em"] for s in run_summaries]
    aggregate = {
        "n_runs": len(run_summaries),
        "ems": ems,
        "em_pct": [round(x * 100, 2) for x in ems],
        "mean_em": statistics.mean(ems) if ems else None,
        "stdev_em": statistics.stdev(ems) if len(ems) > 1 else 0.0,
        "min_em": min(ems) if ems else None,
        "max_em": max(ems) if ems else None,
        "runs": run_summaries,
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print("\n=== aggregate ===", flush=True)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
