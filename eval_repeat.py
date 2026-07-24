#!/usr/bin/env python3
"""Run live_baseline + router eval multiple times and aggregate stability stats."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scipy import stats

from config import EVAL_WORKERS, Config
from eval import run_eval

ROUTER_MODES = {
    "router": "router",
    "ff_router": "ff_router",
}


def _load_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def _paired_stats(base_path: Path, router_path: Path) -> dict:
    base = _load_jsonl(base_path)
    router = _load_jsonl(router_path)
    ids = sorted(set(base) & set(router))
    b = c = 0
    for i in ids:
        be, re = base[i]["em"], router[i]["em"]
        if re and not be:
            b += 1
        elif be and not re:
            c += 1
    p_mcnemar = None
    if b + c:
        p_mcnemar = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {
        "n": len(ids),
        "router_only_correct": b,
        "baseline_only_correct": c,
        "mcnemar_p": p_mcnemar,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--protocol", choices=["native", "paper"], default="native")
    p.add_argument("--router-mode", choices=list(ROUTER_MODES), default="router")
    p.add_argument("--router-url", default="http://127.0.0.1:8084/v1")
    p.add_argument("--router-model", default="qwen3-4b-router")
    p.add_argument("--out-root", default="checkpoints/eval_native_128_repeat")
    p.add_argument("--eval-workers", type=int, default=EVAL_WORKERS)
    p.add_argument(
        "--answer-urls",
        default=None,
        help="comma-separated OSS URLs for parallel reward/eval (default: config ANSWER_URLS)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="solver max_tokens for BOTH live_baseline and router (default: legacy 4k/8k split)",
    )
    args = p.parse_args()

    router_mode = ROUTER_MODES[args.router_mode]
    cfg = Config()
    cfg.router_url = args.router_url
    answer_urls = (
        [u.strip() for u in args.answer_urls.split(",") if u.strip()]
        if args.answer_urls else list(cfg.answer_urls)
    )
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    runs = []
    for i in range(1, args.repeats + 1):
        out_dir = root / f"run{i}"
        print(f"\n=== repeat {i}/{args.repeats} -> {out_dir} ===", flush=True)
        summary = run_eval(
            cfg,
            modes=["live_baseline", router_mode],
            limit=args.limit,
            out_dir=out_dir,
            router_model=args.router_model,
            protocol=args.protocol,
            workers=args.eval_workers,
            answer_urls=answer_urls,
            max_tokens=args.max_tokens,
        )
        router_jsonl = out_dir / f"{router_mode}.jsonl"
        paired = _paired_stats(out_dir / "live_baseline.jsonl", router_jsonl)
        router_em = summary[router_mode]["em"]
        run_row = {
            "run": i,
            "live_baseline_em": summary["live_baseline"]["em"],
            "router_em": router_em,
            "delta_pp": (router_em - summary["live_baseline"]["em"]) * 100,
            **paired,
        }
        runs.append(run_row)
        print(json.dumps(run_row, indent=2), flush=True)

    base_ems = [r["live_baseline_em"] for r in runs]
    router_ems = [r["router_em"] for r in runs]
    deltas = [r["delta_pp"] for r in runs]
    agg = {
        "repeats": args.repeats,
        "limit": args.limit,
        "protocol": args.protocol,
        "router_mode": router_mode,
        "router_model": args.router_model,
        "max_tokens": args.max_tokens,
        "answer_urls": answer_urls,
        "runs": runs,
        "live_baseline_em_mean": sum(base_ems) / len(base_ems),
        "live_baseline_em_std": statistics.pstdev(base_ems) if len(base_ems) > 1 else 0.0,
        "router_em_mean": sum(router_ems) / len(router_ems),
        "router_em_std": statistics.pstdev(router_ems) if len(router_ems) > 1 else 0.0,
        "delta_pp_mean": sum(deltas) / len(deltas),
        "delta_pp_std": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "delta_pp_all_positive": all(d > 0 for d in deltas),
    }
    (root / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print("\n=== aggregate ===", flush=True)
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
