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


def _baseline_one(
    pool: _AnswererPool, cfg: Config, row: dict, protocol: str, *, max_tokens: int,
) -> dict:
    client = pool.next_client()
    rec = rollout_ff(
        client, cfg.answer_model, row["problem"], row["gold"], "",
        protocol=protocol, max_tokens=max_tokens,
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
    max_tokens: int,
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
            small_output=raw, protocol=protocol, max_tokens=max_tokens,
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
    max_tokens: int = 8192,
    only_ids_file: Path | None = None,
) -> dict:
    from core import append_jsonl

    data_file = Path(data_file or cfg.train_file)
    out_dir = Path(out_dir or cfg.ckpt_dir / "oracle_hint")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(data_file)
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
        rows = [r for r in rows if r["id"] in only_ids]
        print(f"only_ids filter: keep {len(rows)}/{len(only_ids)} matched", flush=True)
    if limit:
        rows = rows[:limit]

    # Resume: skip ids that already have a baseline record in this out_dir.
    base_path = out_dir / "baselines.jsonl"
    label_path = out_dir / "oracle_labels.jsonl"
    attempt_path = out_dir / "oracle_attempts.jsonl"
    done_base = set()
    if base_path.exists():
        for rec in load_jsonl(base_path):
            done_base.add(rec["id"])
        if done_base:
            print(f"resume baselines: {len(done_base)} done", flush=True)
    todo_rows = [r for r in rows if r["id"] not in done_base]

    urls = answer_urls or ANSWER_URLS
    pool = _AnswererPool(urls, cfg.answer_model)

    baselines: dict[int, dict] = {}
    if base_path.exists():
        for rec in load_jsonl(base_path):
            baselines[rec["id"]] = rec

    if todo_rows:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_baseline_one, pool, cfg, row, protocol, max_tokens=max_tokens)
                for row in todo_rows
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="baseline"):
                rec = fut.result()
                baselines[rec["id"]] = rec
                append_jsonl(base_path, rec)

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
        f"pool={len(rows)} baselined={len(baselines)} still_wrong_todo={len(wrong_rows)} "
        f"max_tokens={max_tokens} k={k}",
        flush=True,
    )

    all_labels: list[dict] = list(load_jsonl(label_path)) if label_path.exists() else []
    new_labels = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _process_wrong, pool, cfg, row, baselines[row["id"]],
                k=k, hint_temp=hint_temp, protocol=protocol, max_tokens=max_tokens,
            )
            for row in wrong_rows
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="blind-hint"):
            labels, attempts = fut.result()
            for a in attempts:
                append_jsonl(attempt_path, a)
            for lab in labels:
                append_jsonl(label_path, lab)
                all_labels.append(lab)
                new_labels += 1

    n = len(rows)
    n_wrong = sum(1 for row in rows if baselines.get(row["id"], {}).get("em") == 0)
    stats = {
        "n_questions": n,
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
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"questions={n}  baseline_wrong={n_wrong}  labels={len(all_labels)}  "
          f"questions_with_label={stats['n_questions_with_label']}  new={new_labels}")
    print(f"-> {label_path}")
    return stats
