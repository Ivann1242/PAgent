"""Evaluation: baseline, random, router, oracle + precheck."""

from __future__ import annotations

import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config import BASELINE_RES, EVAL_PARQUET, EVAL_WORKERS, Config
from core import (
    ACTION_KEYS,
    ACTION_SPACE,
    build_large_prompt,
    build_optimizer_prompt,
    build_small_prompt,
    call_llm,
    exact_match,
    extract_final_answer,
    load_baseline_predictions,
    load_dapo_rows,
    parse_action,
    parse_optimizer_output,
    random_action,
    rollout,
    rollout_ff,
)


def _metrics(records: list[dict]) -> dict:
    n = max(len(records), 1)
    em = sum(r.get("em", 0) for r in records) / n
    fmt = sum(r.get("format_ok", 0) for r in records) / n
    reward = sum(r.get("reward", 0) for r in records) / n
    invalid = sum(1 for r in records if not r.get("parse_ok", True)) / n
    actions = Counter(r.get("selected_action", "?") for r in records)
    per_action = {}
    for a in ACTION_KEYS:
        subset = [r for r in records if r.get("selected_action") == a]
        if subset:
            per_action[a] = sum(r.get("em", 0) for r in subset) / len(subset)
    return {
        "em": em,
        "format_accuracy": fmt,
        "avg_reward": reward,
        "invalid_action_rate": invalid,
        "action_distribution": dict(actions),
        "per_action_em": per_action,
        "n": len(records),
    }


def eval_baseline(rows: list[dict], baseline: dict[int, str]) -> list[dict]:
    records = []
    for row in rows:
        pred = baseline[row["id"]]
        em = exact_match(pred, row["gold"])
        records.append({
            "id": row["id"], "problem": row["problem"], "gold_answer": row["gold"],
            "selected_action": "baseline", "hint": "", "small_output": "",
            "large_output": "", "pred_answer": pred, "em": em,
            "format_ok": 0, "reward": float(em), "parse_ok": True,
        })
    return records


def eval_random(rows, answer_client, answer_model, *, max_tokens=4096, protocol="native") -> list[dict]:
    records = []
    for row in tqdm(rows, desc="random"):
        action = random_action()
        parse_ok = True
        r = rollout(
            answer_client, answer_model, row["problem"], row["gold"], action,
            parse_ok=parse_ok, max_tokens=max_tokens, protocol=protocol,
        )
        r["id"] = row["id"]
        r["parse_ok"] = parse_ok
        records.append(r)
    return records


ROUTER_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def _run_parallel(
    rows, fn, *, workers: int, desc: str, resume_path: Path | str | None = None,
) -> list[dict]:
    """Run fn over rows; optionally resume/append to resume_path jsonl by id."""
    from core import append_jsonl, load_jsonl

    resume_path = Path(resume_path) if resume_path is not None else None
    done: dict = {}
    if resume_path is not None and resume_path.exists():
        for rec in load_jsonl(resume_path):
            done[rec["id"]] = rec
        if done:
            print(f"[{desc}] resume {resume_path}: {len(done)}/{len(rows)} done", flush=True)

    todo = [row for row in rows if row["id"] not in done]
    if not todo:
        return [done[row["id"]] for row in rows]

    write_lock = threading.Lock()

    def _store(rec: dict) -> None:
        done[rec["id"]] = rec
        if resume_path is not None:
            with write_lock:
                append_jsonl(resume_path, rec)

    if workers <= 1 or len(todo) <= 1:
        for row in tqdm(todo, desc=desc):
            _store(fn(row))
        return [done[row["id"]] for row in rows]

    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
        futures = {pool.submit(fn, row): row["id"] for row in todo}
        for fut in tqdm(as_completed(futures), total=len(todo), desc=desc):
            _store(fut.result())
    return [done[row["id"]] for row in rows]


def _eval_live_baseline_row(row, answer_client, answer_model, *, protocol, max_tokens):
    r = rollout(
        answer_client, answer_model, row["problem"], row["gold"], "baseline",
        max_tokens=max_tokens, protocol=protocol,
    )
    r["id"] = row["id"]
    r["parse_ok"] = True
    return r


def _eval_router_row(row, router_client, answer_client, *, router_model, answer_model,
                     protocol, max_tokens, temperature):
    small_out = call_llm(
        router_client, router_model,
        build_small_prompt(row["problem"]),
        temperature=temperature, max_tokens=128,
        extra_body=ROUTER_EXTRA_BODY,
    )
    action, parse_ok = parse_action(small_out)
    r = rollout(
        answer_client, answer_model, row["problem"], row["gold"], action,
        small_output=small_out, parse_ok=parse_ok, max_tokens=max_tokens,
        protocol=protocol,
    )
    r["id"] = row["id"]
    r["parse_ok"] = parse_ok
    return r


def eval_live_baseline(rows, answer_client, answer_model, *, max_tokens=4096,
                       protocol="native", workers=1, answer_pool=None,
                       resume_path=None) -> list[dict]:
    def fn(row):
        client = answer_pool.next_client() if answer_pool is not None else answer_client
        return _eval_live_baseline_row(
            row, client, answer_model, protocol=protocol, max_tokens=max_tokens,
        )
    return _run_parallel(
        rows, fn, workers=workers, desc="live_baseline", resume_path=resume_path,
    )


def _eval_ff_router_row(row, router_client, answer_client, *, router_model, answer_model,
                        protocol, max_tokens, temperature):
    small_out = call_llm(
        router_client, router_model,
        build_optimizer_prompt(row["problem"]),
        temperature=temperature, max_tokens=256,
        extra_body=ROUTER_EXTRA_BODY,
    )
    hint, parse_ok = parse_optimizer_output(small_out)
    r = rollout_ff(
        answer_client, answer_model, row["problem"], row["gold"], hint,
        small_output=small_out, max_tokens=max_tokens, protocol=protocol,
    )
    r["id"] = row["id"]
    r["parse_ok"] = parse_ok
    return r


def eval_ff_router(rows, router_client, answer_client, *, router_model, answer_model,
                   temperature=0.0, max_tokens=8192, protocol="native", workers=1,
                   answer_pool=None, resume_path=None) -> list[dict]:
    def fn(row):
        client = answer_pool.next_client() if answer_pool is not None else answer_client
        return _eval_ff_router_row(
            row, router_client, client,
            router_model=router_model, answer_model=answer_model,
            protocol=protocol, max_tokens=max_tokens, temperature=temperature,
        )
    return _run_parallel(
        rows, fn, workers=workers, desc="ff_router", resume_path=resume_path,
    )


def eval_router(rows, router_client, answer_client, *, router_model, answer_model,
                temperature=0.0, max_tokens=4096, protocol="native", workers=1) -> list[dict]:
    fn = lambda row: _eval_router_row(
        row, router_client, answer_client,
        router_model=router_model, answer_model=answer_model,
        protocol=protocol, max_tokens=max_tokens, temperature=temperature,
    )
    return _run_parallel(rows, fn, workers=workers, desc="router")


def eval_per_action(rows, answer_client, answer_model, *, max_tokens=4096, protocol="native") -> dict[str, list[dict]]:
    """Per-action EM for headroom check."""
    by_action: dict[str, list[dict]] = {a: [] for a in ACTION_KEYS}
    for row in tqdm(rows, desc="per-action"):
        for action in ACTION_KEYS:
            r = rollout(
                answer_client, answer_model, row["problem"], row["gold"], action,
                max_tokens=max_tokens, protocol=protocol,
            )
            r["id"] = row["id"]
            by_action[action].append(r)
    return by_action


def eval_oracle(rows, answer_client, answer_model, *, max_tokens=4096, protocol="native") -> list[dict]:
    records = []
    for row in tqdm(rows, desc="oracle"):
        best = None
        for action in ACTION_KEYS:
            r = rollout(
                answer_client, answer_model, row["problem"], row["gold"], action,
                max_tokens=max_tokens, protocol=protocol,
            )
            if best is None or r["em"] > best["em"] or (
                r["em"] == best["em"] and r["reward"] > best["reward"]
            ):
                best = {**r, "selected_action": action, "id": row["id"], "parse_ok": True}
        records.append(best)
    return records


def run_precheck(cfg: Config, *, limit=16, out_dir=None) -> dict:
    """Per-action + oracle eval; warn if headroom < 1%."""
    out_dir = Path(out_dir or cfg.ckpt_dir / "precheck")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dapo_rows(EVAL_PARQUET)[:limit]
    baseline = load_baseline_predictions(BASELINE_RES)
    answer_client = OpenAI(base_url=cfg.answer_url, api_key="EMPTY")

    base_recs = eval_baseline(rows, baseline)
    base_em = _metrics(base_recs)["em"]

    by_action = eval_per_action(rows, answer_client, cfg.answer_model)
    per_action_em = {a: _metrics(recs)["em"] for a, recs in by_action.items()}
    (out_dir / "per_action.json").write_text(json.dumps(per_action_em, indent=2))

    oracle_recs = eval_oracle(rows, answer_client, cfg.answer_model)
    oracle_em = _metrics(oracle_recs)["em"]
    headroom = oracle_em - base_em

    report = {
        "baseline_em": base_em,
        "oracle_em": oracle_em,
        "headroom": headroom,
        "per_action_em": per_action_em,
        "pass": headroom >= 0.01,
    }
    (out_dir / "precheck.json").write_text(json.dumps(report, indent=2))

    print(f"baseline EM: {base_em:.1%}")
    print(f"oracle EM:   {oracle_em:.1%}")
    print(f"headroom:    {headroom:+.1%}")
    for a, em in per_action_em.items():
        print(f"  {a}: {em:.1%}")
    if not report["pass"]:
        print("WARNING: headroom < 1%, action space may lack signal for GRPO")
    return report


def run_eval(cfg: Config, *, modes=None, limit=None, out_dir=None,
             router_model=None, protocol="native", workers=EVAL_WORKERS,
             answer_urls: list[str] | None = None,
             max_tokens: int | None = None) -> dict:
    modes = modes or ["baseline", "random", "router", "oracle"]
    if out_dir is None:
        out_dir = cfg.ckpt_dir / ("eval_paper" if protocol == "paper" else "eval")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dapo_rows(EVAL_PARQUET)
    if limit:
        rows = rows[:limit]
    baseline = load_baseline_predictions(BASELINE_RES)
    urls = list(answer_urls or cfg.answer_urls or [cfg.answer_url])
    answer_client = OpenAI(base_url=urls[0], api_key="EMPTY")
    answer_pool = None
    if len(urls) > 1:
        from label import _AnswererPool
        answer_pool = _AnswererPool(urls, cfg.answer_model)
        print(f"eval answer pool: {len(urls)} urls, workers={workers}", flush=True)
    router_client = OpenAI(base_url=cfg.router_url, api_key="EMPTY")
    router_model = router_model or cfg.router_model

    # Defaults historically differed (baseline 4k vs ff 8k); pass max_tokens to align.
    baseline_tokens = 4096 if max_tokens is None else max_tokens
    ff_tokens = 8192 if max_tokens is None else max_tokens
    router_tokens = 4096 if max_tokens is None else max_tokens
    results = {
        "protocol": protocol,
        "answer_urls": urls,
        "max_tokens": {
            "live_baseline": baseline_tokens,
            "ff_router": ff_tokens,
            "router": router_tokens,
        },
    }
    if "baseline" in modes:
        recs = eval_baseline(rows, baseline)
        results["baseline"] = _metrics(recs)
        (out_dir / "baseline.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n"
        )

    if "random" in modes:
        recs = eval_random(
            rows, answer_client, cfg.answer_model, protocol=protocol,
            max_tokens=baseline_tokens,
        )
        results["random"] = _metrics(recs)

    if "live_baseline" in modes:
        recs = eval_live_baseline(
            rows, answer_client, cfg.answer_model, protocol=protocol, workers=workers,
            answer_pool=answer_pool, max_tokens=baseline_tokens,
        )
        results["live_baseline"] = _metrics(recs)
        (out_dir / "live_baseline.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n"
        )

    if "router" in modes:
        recs = eval_router(
            rows, router_client, answer_client,
            router_model=router_model, answer_model=cfg.answer_model,
            protocol=protocol, workers=workers, max_tokens=router_tokens,
        )
        results["router"] = _metrics(recs)
        (out_dir / "router.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n"
        )

    if "ff_router" in modes:
        recs = eval_ff_router(
            rows, router_client, answer_client,
            router_model=router_model or cfg.router_model, answer_model=cfg.answer_model,
            protocol=protocol, workers=workers, answer_pool=answer_pool,
            max_tokens=ff_tokens,
        )
        results["ff_router"] = _metrics(recs)
        (out_dir / "ff_router.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n"
        )

    if "oracle" in modes:
        recs = eval_oracle(rows, answer_client, cfg.answer_model, protocol=protocol)
        results["oracle"] = _metrics(recs)

    summary_path = out_dir / "summary.json"
    merged = {}
    if summary_path.exists():
        merged = json.loads(summary_path.read_text())
    merged.update(results)

    if "baseline" in merged and "router" in merged:
        merged["router_vs_baseline"] = merged["router"]["em"] - merged["baseline"]["em"]
    if "live_baseline" in merged and "router" in merged:
        merged["router_vs_live_baseline"] = (
            merged["router"]["em"] - merged["live_baseline"]["em"]
        )
    if "random" in merged and "router" in merged:
        merged["router_vs_random"] = merged["router"]["em"] - merged["random"]["em"]
    if "live_baseline" in merged and "ff_router" in merged:
        merged["ff_router_vs_live_baseline"] = (
            merged["ff_router"]["em"] - merged["live_baseline"]["em"]
        )
    if "random" in merged and "ff_router" in merged:
        merged["ff_router_vs_random"] = merged["ff_router"]["em"] - merged["random"]["em"]

    summary_path.write_text(json.dumps(merged, indent=2))
    print(json.dumps(merged, indent=2))
    return merged
