"""Exhaustive per-action rollout for supervised labeling."""

from __future__ import annotations

import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config import ANSWER_URLS, EVAL_WORKERS, Config
from core import ACTION_KEYS, load_jsonl, rollout, write_jsonl


def classify_action_ems(action_ems: dict[str, int]) -> tuple[str, list[str]]:
    """Return (category, best_actions). category in signal|all_correct|all_wrong."""
    ems = list(action_ems.values())
    if all(e == 0 for e in ems):
        return "all_wrong", []
    if all(e == 1 for e in ems):
        return "all_correct", list(action_ems)
    if min(ems) != max(ems):
        best = [a for a, em in action_ems.items() if em == 1]
        if best:
            return "signal", best
    return "all_wrong", []


class _AnswererPool:
    """Round-robin OpenAI clients across multiple vLLM answerer endpoints."""

    def __init__(self, urls: list[str], model: str):
        self._clients = cycle([OpenAI(base_url=u, api_key="EMPTY") for u in urls])
        self._lock = threading.Lock()
        self.model = model
        self.urls = urls

    def next_client(self) -> OpenAI:
        with self._lock:
            return next(self._clients)


def run_label(
    cfg: Config,
    *,
    limit: int | None = None,
    data_file: Path | None = None,
    out_dir: Path | None = None,
    workers: int = EVAL_WORKERS,
    answer_urls: list[str] | None = None,
    protocol: str = "native",
) -> dict:
    data_file = Path(data_file or cfg.train_file)
    out_dir = Path(out_dir or cfg.ckpt_dir / "label_pilot")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(data_file)
    if limit:
        rows = rows[:limit]

    urls = answer_urls or ANSWER_URLS
    pool = _AnswererPool(urls, cfg.answer_model)
    tasks = [(row, action) for row in rows for action in ACTION_KEYS]
    rollouts: dict[int, dict[str, dict]] = {row["id"]: {} for row in rows}

    def _one(task):
        row, action = task
        client = pool.next_client()
        r = rollout(
            client, cfg.answer_model, row["problem"], row["gold"], action,
            protocol=protocol,
        )
        r["id"] = row["id"]
        return row["id"], action, r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="label-rollout"):
            qid, action, rec = fut.result()
            rollouts[qid][action] = rec

    all_records: list[dict] = []
    labels: list[dict] = []
    categories = Counter()

    for row in rows:
        qid = row["id"]
        action_recs = rollouts[qid]
        action_ems = {a: action_recs[a]["em"] for a in ACTION_KEYS}
        category, best_actions = classify_action_ems(action_ems)
        categories[category] += 1

        group = {
            "id": qid,
            "problem": row["problem"],
            "gold": row["gold"],
            "router_prompt": row.get("router_prompt"),
            "action_ems": action_ems,
            "category": category,
            "best_actions": best_actions,
            "label_action": best_actions[0] if best_actions else None,
            "rollouts": action_recs,
        }
        all_records.append(group)
        if category == "signal":
            labels.append({
                "id": qid,
                "problem": row["problem"],
                "gold": row["gold"],
                "router_prompt": row.get("router_prompt"),
                "label_action": best_actions[0],
                "best_actions": best_actions,
                "action_ems": action_ems,
            })

    n = len(rows)
    stats = {
        "n_questions": n,
        "n_rollouts": len(tasks),
        "n_actions": len(ACTION_KEYS),
        "categories": dict(categories),
        "signal_rate": categories["signal"] / n if n else 0.0,
        "all_correct_rate": categories["all_correct"] / n if n else 0.0,
        "all_wrong_rate": categories["all_wrong"] / n if n else 0.0,
        "n_labels": len(labels),
        "label_action_distribution": dict(Counter(l["label_action"] for l in labels)),
        "per_action_em": {
            a: sum(rollouts[qid][a]["em"] for qid in rollouts) / n if n else 0.0
            for a in ACTION_KEYS
        },
        "data_file": str(data_file),
        "answer_urls": urls,
        "workers": workers,
        "protocol": protocol,
    }

    write_jsonl(out_dir / "rollouts.jsonl", all_records)
    write_jsonl(out_dir / "labels.jsonl", labels)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    print(f"questions: {n}  rollouts: {len(tasks)}  workers: {workers}  endpoints: {len(urls)}")
    print(f"signal:      {categories['signal']:4d}  ({100*stats['signal_rate']:.1f}%)  -> {len(labels)} labels")
    print(f"all_correct: {categories['all_correct']:4d}  ({100*stats['all_correct_rate']:.1f}%)")
    print(f"all_wrong:   {categories['all_wrong']:4d}  ({100*stats['all_wrong_rate']:.1f}%)")
    print("label actions:", stats["label_action_distribution"])
    print(f"-> {out_dir / 'stats.json'}")
    return stats
