#!/usr/bin/env python3
"""Evaluate FF router on in-distribution (training-label) questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import EVAL_WORKERS, Config
from core import load_jsonl, make_openai_client, rollout_ff, write_jsonl
from eval import _metrics, eval_ff_router, eval_live_baseline


def load_idist_rows(labels_file: Path) -> list[dict]:
    """One row per question id covered by the label file."""
    by_id: dict[int, dict] = {}
    for row in load_jsonl(labels_file):
        qid = row["id"]
        if qid not in by_id:
            by_id[qid] = {
                "id": qid,
                "problem": row["problem"],
                "gold": row.get("gold") or row.get("gold_answer", ""),
                "label_hint": row.get("label_hint", ""),
            }
    return [by_id[i] for i in sorted(by_id)]


def eval_oracle_hint(
    rows, answer_client, answer_model, *, protocol="native", workers=1, max_tokens=8192,
    answer_pool=None, resume_path=None,
):
    from eval import _run_parallel

    def _one(row):
        client = answer_pool.next_client() if answer_pool is not None else answer_client
        r = rollout_ff(
            client, answer_model, row["problem"], row["gold"], row["label_hint"],
            protocol=protocol, max_tokens=max_tokens,
        )
        r["id"] = row["id"]
        r["selected_action"] = "oracle_hint"
        return r

    return _run_parallel(
        rows, _one, workers=workers, desc="oracle_hint", resume_path=resume_path,
    )


def run_idist_eval(
    cfg: Config,
    *,
    labels_file: Path,
    out_dir: Path,
    router_model: str,
    router_url: str | None = None,
    answer_urls: list[str] | None = None,
    modes: list[str] | None = None,
    limit: int | None = None,
    workers: int = EVAL_WORKERS,
    protocol: str = "native",
    max_tokens: int = 8192,
) -> dict:
    modes = modes or ["live_baseline", "ff_router", "oracle_hint"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_idist_rows(labels_file)
    if limit:
        rows = rows[:limit]

    urls = list(answer_urls or cfg.answer_urls or [cfg.answer_url])
    answer_client = make_openai_client(urls[0])
    answer_pool = None
    if len(urls) > 1:
        from label import _AnswererPool
        answer_pool = _AnswererPool(urls, cfg.answer_model)
        print(f"idist answer pool: {len(urls)} urls, workers={workers}", flush=True)
    router_client = make_openai_client(router_url or cfg.router_url)

    meta = {
        "labels_file": str(labels_file),
        "n_questions": len(rows),
        "router_model": router_model,
        "router_url": router_url or cfg.router_url,
        "answer_urls": urls,
        "protocol": protocol,
        "modes": modes,
        "max_tokens": max_tokens,
    }
    results = {"meta": meta}

    if "live_baseline" in modes:
        path = out_dir / "live_baseline.jsonl"
        recs = eval_live_baseline(
            rows, answer_client, cfg.answer_model,
            protocol=protocol, workers=workers, max_tokens=max_tokens,
            answer_pool=answer_pool, resume_path=path,
        )
        results["live_baseline"] = _metrics(recs)
        write_jsonl(path, recs)

    if "ff_router" in modes:
        path = out_dir / "ff_router.jsonl"
        recs = eval_ff_router(
            rows, router_client, answer_client,
            router_model=router_model, answer_model=cfg.answer_model,
            protocol=protocol, workers=workers, max_tokens=max_tokens,
            answer_pool=answer_pool, resume_path=path,
        )
        results["ff_router"] = _metrics(recs)
        write_jsonl(path, recs)

    if "oracle_hint" in modes:
        path = out_dir / "oracle_hint.jsonl"
        recs = eval_oracle_hint(
            rows, answer_client, cfg.answer_model,
            protocol=protocol, workers=workers, max_tokens=max_tokens,
            answer_pool=answer_pool, resume_path=path,
        )
        results["oracle_hint"] = _metrics(recs)
        write_jsonl(path, recs)

    if "live_baseline" in results and "ff_router" in results:
        results["ff_router_vs_live_baseline"] = (
            results["ff_router"]["em"] - results["live_baseline"]["em"]
        )
    if "oracle_hint" in results and "ff_router" in results:
        results["ff_router_vs_oracle_hint"] = (
            results["ff_router"]["em"] - results["oracle_hint"]["em"]
        )

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--router-model", required=True)
    p.add_argument("--router-url", default="http://127.0.0.1:8086/v1")
    p.add_argument(
        "--answer-urls",
        default=None,
        help="comma-separated OSS OpenAI base urls (enables round-robin pool)",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=EVAL_WORKERS)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--protocol", choices=["native", "paper"], default="native")
    p.add_argument(
        "--modes", nargs="+",
        default=["live_baseline", "ff_router", "oracle_hint"],
    )
    args = p.parse_args()
    answer_urls = (
        [u.strip() for u in args.answer_urls.split(",") if u.strip()]
        if args.answer_urls
        else None
    )

    run_idist_eval(
        Config(),
        labels_file=Path(args.labels_file),
        out_dir=Path(args.out_dir),
        router_model=args.router_model,
        router_url=args.router_url,
        answer_urls=answer_urls,
        modes=args.modes,
        limit=args.limit,
        workers=args.workers,
        protocol=args.protocol,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
