#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from core import append_jsonl, load_jsonl, write_jsonl
from HintFlow_five.agent import FiveTurnAgent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="checkpoints/hintflow_five_data/iid_512.jsonl")
    p.add_argument("--out-dir", default="checkpoints/eval_hintflow_five_iid512")
    p.add_argument("--mode", choices=["baseline", "router", "oracle"], required=True)
    p.add_argument("--router-url", default="http://127.0.0.1:8086/v1")
    p.add_argument("--router-model", default="hintflow-five-router")
    p.add_argument("--selector-url", default=None)
    p.add_argument("--selector-model", default=None)
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--solver-urls", default="http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1")
    p.add_argument("--solver-model", default="qwen3-14b")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--solver-max-tokens", type=int, default=2048)
    p.add_argument("--selector-max-tokens", type=int, default=192)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260817)
    args = p.parse_args()
    rows = load_jsonl(Path(args.data))
    if args.limit is not None:
        rows = rows[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.mode}.jsonl"
    done = {r["id"]: r for r in load_jsonl(out_file) if not r.get("error")} if out_file.exists() else {}
    urls = [x.strip() for x in args.solver_urls.split(",") if x.strip()]
    local = threading.local()
    lock = threading.Lock()

    def work(row):
        solver_url = urls[int(row["id"]) % len(urls)]
        agents = getattr(local, "agents", None)
        if agents is None:
            agents = local.agents = {}
        key = (solver_url, args.mode)
        router_url = solver_url if args.mode == "oracle" else args.router_url
        router_model = args.solver_model if args.mode == "oracle" else args.router_model
        agent = agents.setdefault(key, FiveTurnAgent(
            router_url=router_url,
            router_model=router_model,
            solver_url=solver_url,
            solver_model=args.solver_model,
            selector_url=args.selector_url or args.router_url,
            selector_model=args.selector_model or args.router_model,
            n_turns=args.turns,
            solver_max_tokens=args.solver_max_tokens,
            selector_max_tokens=args.selector_max_tokens,
            seed=args.seed + int(row["id"]) * 100,
        ))
        try:
            rec = agent.run(row["problem"], gold=row["gold"], mode=args.mode).to_dict()
            rec.update({"id": row["id"], "solver_url": solver_url})
            return rec
        except Exception as exc:
            return {"id": row["id"], "problem": row["problem"], "gold": row["gold"],
                    "mode": args.mode, "final_answer": "", "em": 0, "turns": [],
                    "solver_url": solver_url, "error": f"{type(exc).__name__}: {exc}"}

    todo = [r for r in rows if r["id"] not in done]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(todo)))) as pool:
        futures = {pool.submit(work, row): row for row in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=args.mode):
            rec = fut.result()
            if not rec.get("error"):
                done[rec["id"]] = rec
                with lock:
                    append_jsonl(out_file, rec)
            else:
                print(f"id={rec['id']} {rec['error']}", flush=True)
    records = [done[r["id"]] for r in rows if r["id"] in done]
    write_jsonl(out_file, records)
    decisions = Counter(
        turn["decision"] for rec in records for turn in rec.get("turns", [])
    )
    parse_fail = sum(
        not turn.get("selector_parse_ok", True)
        for rec in records for turn in rec.get("turns", [])
    )
    summary = {
        "mode": args.mode,
        "n": len(records),
        "requested_n": len(rows),
        "em": sum(int(r.get("em") or 0) for r in records) / max(len(records), 1),
        "decisions": dict(decisions),
        "selector_parse_failures": parse_fail,
        "errors_or_missing": len(rows) - len(records),
        "elapsed_sec": round(time.time() - t0, 1),
        "router_model": args.router_model if args.mode == "router" else None,
        "selector_model": args.selector_model or args.router_model,
        "n_turns": args.turns,
        "solver_model": args.solver_model,
    }
    (out_dir / f"{args.mode}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if len(records) != len(rows):
        raise SystemExit("evaluation incomplete; rerun to resume")


if __name__ == "__main__":
    main()

