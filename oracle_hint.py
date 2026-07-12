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


def _baseline_one(pool: _AnswererPool, cfg: Config, row: dict, protocol: str) -> dict:
    client = pool.next_client()
    rec = rollout_ff(
        client, cfg.answer_model, row["problem"], row["gold"], "",
        protocol=protocol,
    )
    rec["id"] = row["id"]
    return rec


def _process_wrong(
    pool: _AnswererPool,
    cfg: Config,
    row: dict,
    baseline_rec: dict,
    *,
    k: int,
    hint_temp: float,
    protocol: str,
) -> tuple[list[dict], list[dict]]:
    labels: list[dict] = []
    attempts: list[dict] = []
    seen: set[str] = set()

    for _ in range(k):
        gen_client = pool.next_client()
        raw = call_llm(
            gen_client, cfg.answer_model,
            build_optimizer_prompt(row["problem"]),
            temperature=hint_temp, max_tokens=384,
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
            })
            continue
        seen.add(key)

        ans_client = pool.next_client()
        rec = rollout_ff(
            ans_client, cfg.answer_model, row["problem"], row["gold"], hint,
            small_output=raw, protocol=protocol,
        )
        attempts.append({
            "id": row["id"],
            "stage": "hint_test",
            "problem": row["problem"],
            "gold": row["gold"],
            "hint": hint,
            "gen_output": raw,
            "baseline_em": baseline_rec["em"],
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
) -> dict:
    data_file = Path(data_file or cfg.train_file)
    out_dir = Path(out_dir or cfg.ckpt_dir / "oracle_hint")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(data_file)
    if limit:
        rows = rows[:limit]

    urls = answer_urls or ANSWER_URLS
    pool = _AnswererPool(urls, cfg.answer_model)

    baselines: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_baseline_one, pool, cfg, row, protocol) for row in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="baseline"):
            rec = fut.result()
            baselines[rec["id"]] = rec

    wrong_rows = [row for row in rows if baselines[row["id"]]["em"] == 0]
    write_jsonl(out_dir / "baselines.jsonl", [baselines[row["id"]] for row in rows])

    all_labels: list[dict] = []
    all_attempts: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _process_wrong, pool, cfg, row, baselines[row["id"]],
                k=k, hint_temp=hint_temp, protocol=protocol,
            )
            for row in wrong_rows
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="blind-hint"):
            labels, attempts = fut.result()
            all_labels.extend(labels)
            all_attempts.extend(attempts)

    write_jsonl(out_dir / "oracle_labels.jsonl", all_labels)
    write_jsonl(out_dir / "oracle_attempts.jsonl", all_attempts)

    n = len(rows)
    stats = {
        "n_questions": n,
        "n_baseline_wrong": len(wrong_rows),
        "n_hint_candidates": sum(1 for a in all_attempts if a.get("stage") == "hint_test"),
        "n_labels": len(all_labels),
        "n_questions_with_label": len({l["id"] for l in all_labels}),
        "label_rate": len({l["id"] for l in all_labels}) / n if n else 0.0,
        "hint_mode": "blind",
        "k": k,
        "hint_temp": hint_temp,
        "data_file": str(data_file),
        "answer_urls": urls,
        "workers": workers,
        "protocol": protocol,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"questions={n}  baseline_wrong={len(wrong_rows)}  labels={len(all_labels)}  "
          f"questions_with_label={stats['n_questions_with_label']}")
    print(f"-> {out_dir / 'oracle_labels.jsonl'}")
    return stats
