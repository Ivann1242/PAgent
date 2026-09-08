"""Blind hint labeling: OSS generates hints from problem only; keep baseline-wrong -> hint-correct."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from config import ANSWER_URLS, EVAL_WORKERS, Config
from core import (
    build_optimizer_prompt,
    call_llm,
    hint_leaks_gold,
    load_jsonl,
    parse_optimizer_output,
    rollout_ff,
    write_jsonl,
)
from label import _AnswererPool

_META_PREFIXES = ("we need to", "we must", "to solve,")


def _looks_like_meta_hint(hint: str) -> bool:
    h = hint.strip().lower()
    return h.startswith(_META_PREFIXES) or len(h) > 600


def _route_key(qid: int, sticky_mode: str) -> int:
    """Map question id → sticky routing key.

    - id: id % n_urls (default)
    - id_div4: (id // 4) % n_urls — spreads a single id%4 residue class across GPUs
    - hash: multiplicative hash % n_urls — spreads arbitrary id subsets evenly
      (needed when remaining backlog is itself one id_div4 bucket).
    """
    qid = int(qid)
    if sticky_mode == "id_div4":
        return qid // 4
    if sticky_mode == "hash":
        # Avalanche mix — plain Knuth hash collides on arithmetic progressions
        # like id ≡ 15 (mod 16), which is exactly a crashed id_div4 bucket.
        x = qid & 0xFFFFFFFF
        x = ((x >> 16) ^ x) * 0x45D9F3B
        x &= 0xFFFFFFFF
        x = ((x >> 16) ^ x) * 0x45D9F3B
        x &= 0xFFFFFFFF
        return (x >> 16) ^ x
    return qid


def _baseline_one(
    pool: _AnswererPool, cfg: Config, row: dict, protocol: str, *,
    max_tokens: int, base_seed: int, sticky_mode: str = "id",
) -> dict:
    # Sticky shard: same id → same GPU for cross-run reproducibility.
    rkey = _route_key(row["id"], sticky_mode)
    client = pool.client_for_key(rkey)
    seed = int(base_seed) + int(row["id"])
    url = pool.url_for_key(rkey)
    try:
        rec = rollout_ff(
            client, cfg.answer_model, row["problem"], row["gold"], "",
            protocol=protocol, max_tokens=max_tokens, seed=seed,
        )
        rec["id"] = row["id"]
        rec["seed"] = seed
        rec["solver_url"] = url
        rec["error"] = None
        return rec
    except Exception as e:  # noqa: BLE001
        return {
            "id": row["id"],
            "problem": row["problem"],
            "gold_answer": row["gold"],
            "selected_action": "freeform",
            "hint": "",
            "small_output": "",
            "large_output": "",
            "pred_answer": "",
            "em": 0,
            "format_ok": 0,
            "reward": 0.0,
            "protocol": protocol,
            "mode": "freeform",
            "seed": seed,
            "solver_url": url,
            "error": f"{type(e).__name__}: {e}",
        }


def _process_wrong(
    pool: _AnswererPool,
    cfg: Config,
    row: dict,
    baseline_rec: dict,
    *,
    k: int,
    hint_temp: float,
    protocol: str,
    max_tokens: int,
    base_seed: int,
    sticky_mode: str = "id",
) -> tuple[list[dict], list[dict]]:
    labels: list[dict] = []
    attempts: list[dict] = []
    seen: set[str] = set()
    qid = int(row["id"])
    rkey = _route_key(qid, sticky_mode)
    client = pool.client_for_key(rkey)
    url = pool.url_for_key(rkey)

    for attempt_i in range(k):
        gen_seed = int(base_seed) + qid * 1000 + attempt_i
        raw = call_llm(
            client, cfg.answer_model,
            build_optimizer_prompt(row["problem"]),
            temperature=hint_temp, max_tokens=384, seed=gen_seed,
        )
        hint, parse_ok = parse_optimizer_output(raw)
        key = hint.strip().lower()
        if (
            not parse_ok or not hint or key in seen
            or hint_leaks_gold(hint, row["gold"])
            or _looks_like_meta_hint(hint)
        ):
            attempts.append({
                "id": row["id"], "stage": "gen_skip",
                "hint": hint, "parse_ok": parse_ok, "gen_output": raw,
                "seed": gen_seed, "attempt": attempt_i,
                "solver_url": url,
            })
            continue
        seen.add(key)

        solve_seed = int(base_seed) + qid * 1000 + 500 + attempt_i
        rec = rollout_ff(
            client, cfg.answer_model, row["problem"], row["gold"], hint,
            small_output=raw, protocol=protocol, max_tokens=max_tokens,
            seed=solve_seed,
        )
        attempts.append({
            "id": row["id"],
            "stage": "hint_test",
            "problem": row["problem"],
            "gold": row["gold"],
            "hint": hint,
            "gen_output": raw,
            "baseline_em": baseline_rec["em"],
            "seed_gen": gen_seed,
            "seed_solve": solve_seed,
            "attempt": attempt_i,
            "solver_url": url,
            **rec,
        })

        if baseline_rec["em"] == 0 and rec["em"] == 1:
            labels.append({
                "id": row["id"],
                "problem": row["problem"],
                "gold": row["gold"],
                "label_hint": hint,
                "baseline_em": 0,
                "hint_em": 1,
                "gen_output": raw,
                "seed_gen": gen_seed,
                "seed_solve": solve_seed,
                "attempt": attempt_i,
                "solver_url": url,
            })

    return labels, attempts


def run_oracle_hint(
    cfg: Config,
    *,
    limit: int | None = None,
    data_file: Path | None = None,
    out_dir: Path | None = None,
    workers: int = EVAL_WORKERS,
    answer_urls: list[str] | None = None,
    protocol: str = "native",
    k: int = 6,
    hint_temp: float = 0.8,
    max_tokens: int = 8192,
    only_ids_file: Path | None = None,
    base_seed: int = 42,
    sticky_mode: str = "id",
) -> dict:
    from core import append_jsonl

    data_file = Path(data_file or cfg.train_file)
    out_dir = Path(out_dir or cfg.ckpt_dir / "oracle_hint")
    out_dir.mkdir(parents=True, exist_ok=True)
    if sticky_mode not in {"id", "id_div4", "hash"}:
        raise ValueError(f"sticky_mode must be id|id_div4|hash, got {sticky_mode!r}")

    all_rows = load_jsonl(data_file)
    rows = all_rows
    if only_ids_file is not None:
        # support jsonl {"id": ...} or plain id-per-line
        only_ids: set = set()
        for line in open(only_ids_file):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                only_ids.add(json.loads(line)["id"])
            else:
                only_ids.add(int(line))
        rows = [r for r in all_rows if r["id"] in only_ids]
        print(f"only_ids filter: keep {len(rows)}/{len(only_ids)} matched", flush=True)
    if limit:
        rows = rows[:limit]

    # Resume: skip ids that already have a successful baseline in this out_dir.
    base_path = out_dir / "baselines.jsonl"
    label_path = out_dir / "oracle_labels.jsonl"
    attempt_path = out_dir / "oracle_attempts.jsonl"

    baselines: dict[int, dict] = {}
    if base_path.exists():
        for rec in load_jsonl(base_path):
            if rec.get("error"):
                continue
            baselines[rec["id"]] = rec
        if baselines:
            print(f"resume baselines: {len(baselines)} done", flush=True)
    todo_rows = [r for r in rows if r["id"] not in baselines]

    urls = answer_urls or ANSWER_URLS
    from core import LLM_CONNECT_TIMEOUT_S, LLM_LABEL_TIMEOUT_S

    pool = _AnswererPool(
        urls, cfg.answer_model,
        timeout=LLM_LABEL_TIMEOUT_S,
        connect_timeout=LLM_CONNECT_TIMEOUT_S,
    )

    n_fail = 0
    if todo_rows:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _baseline_one, pool, cfg, row, protocol,
                    max_tokens=max_tokens, base_seed=base_seed,
                    sticky_mode=sticky_mode,
                )
                for row in todo_rows
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="baseline"):
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    n_fail += 1
                    print(f"[baseline] worker crash: {type(e).__name__}: {e}", flush=True)
                    continue
                if rec.get("error"):
                    n_fail += 1
                    # Do not persist — leave id unfinished for resume after shard recovery.
                    continue
                baselines[rec["id"]] = rec
                append_jsonl(base_path, rec)
        if n_fail:
            print(f"[baseline] transient failures (will resume): {n_fail}", flush=True)

    # Hint only for still-wrong ids not yet processed (any attempt row counts).
    hinted_ids = set()
    if attempt_path.exists():
        for a in load_jsonl(attempt_path):
            hinted_ids.add(a["id"])

    wrong_rows = [
        row for row in rows
        if baselines.get(row["id"], {}).get("em") == 0 and row["id"] not in hinted_ids
    ]
    print(
        f"pool={len(rows)} baselined_in_scope={sum(1 for r in rows if r['id'] in baselines)} "
        f"still_wrong_todo={len(wrong_rows)} max_tokens={max_tokens} k={k} "
        f"sticky_mode={sticky_mode} urls={len(urls)} seed={base_seed} workers={workers}",
        flush=True,
    )

    new_labels = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _process_wrong, pool, cfg, row, baselines[row["id"]],
                k=k, hint_temp=hint_temp, protocol=protocol, max_tokens=max_tokens,
                base_seed=base_seed, sticky_mode=sticky_mode,
            )
            for row in wrong_rows
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="blind-hint"):
            try:
                labels, attempts = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[blind-hint] worker crash: {type(e).__name__}: {e}", flush=True)
                continue
            for a in attempts:
                append_jsonl(attempt_path, a)
            for lab in labels:
                append_jsonl(label_path, lab)
                new_labels += 1

    # Full-corpus stats (not just only_ids scope) so backfills don't clobber totals.
    all_labels = list(load_jsonl(label_path)) if label_path.exists() else []
    full_baselines = {
        rec["id"]: rec
        for rec in (load_jsonl(base_path) if base_path.exists() else [])
        if not rec.get("error")
    }
    n_full = len(all_rows) if only_ids_file is not None else len(rows)
    # When not filtering ids, rows may be limit-truncated; use that universe.
    universe = all_rows if only_ids_file is not None else rows
    n = len(universe)
    n_wrong = sum(1 for row in universe if full_baselines.get(row["id"], {}).get("em") == 0)
    n_base = sum(1 for row in universe if row["id"] in full_baselines)
    stats = {
        "n_questions": n,
        "n_baselined": n_base,
        "n_baseline_missing": n - n_base,
        "n_baseline_wrong": n_wrong,
        "n_hint_candidates": sum(
            1 for a in (load_jsonl(attempt_path) if attempt_path.exists() else [])
            if a.get("stage") == "hint_test"
        ),
        "n_labels": len(all_labels),
        "n_questions_with_label": len({l["id"] for l in all_labels}),
        "label_rate": len({l["id"] for l in all_labels}) / n if n else 0.0,
        "new_labels_this_run": new_labels,
        "hint_mode": "blind",
        "k": k,
        "hint_temp": hint_temp,
        "max_tokens": max_tokens,
        "only_ids_file": str(only_ids_file) if only_ids_file else None,
        "data_file": str(data_file),
        "answer_urls": urls,
        "workers": workers,
        "protocol": protocol,
        "base_seed": base_seed,
        "routing": f"sticky_{sticky_mode}",
        "sticky_mode": sticky_mode,
        "answer_model": cfg.answer_model,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    if only_ids_file is not None:
        (out_dir / "stats_backfill.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(
        f"questions={n} baselined={n_base} missing={n - n_base} baseline_wrong={n_wrong} "
        f"labels={len(all_labels)} questions_with_label={stats['n_questions_with_label']} "
        f"new={new_labels}",
        flush=True,
    )
    print(f"-> {label_path}")
    return stats
