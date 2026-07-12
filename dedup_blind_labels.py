"""Pick one flip hint per question via repeated OSS verification."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from config import ANSWER_URLS, EVAL_WORKERS, Config
from core import load_jsonl, rollout_ff, write_jsonl
from label import _AnswererPool


def _hint_key(hint: str) -> str:
    return " ".join(hint.strip().lower().split())


def _group_candidates(rows: list[dict]) -> dict[str, dict]:
    """Map hint_key -> representative label row (first seen)."""
    out: dict[str, dict] = {}
    for row in rows:
        key = _hint_key(row["label_hint"])
        if key not in out:
            out[key] = row
    return out


def _score_hint(
    pool: _AnswererPool,
    cfg: Config,
    row: dict,
    *,
    repeats: int,
    protocol: str,
) -> tuple[float, list[int]]:
    ems: list[int] = []
    for _ in range(repeats):
        client = pool.next_client()
        rec = rollout_ff(
            client, cfg.answer_model, row["problem"], row["gold"], row["label_hint"],
            protocol=protocol,
        )
        ems.append(int(rec["em"]))
    return sum(ems) / len(ems), ems


def dedup_blind_labels(
    cfg: Config,
    *,
    labels_file: Path,
    out_file: Path,
    repeats: int = 3,
    workers: int = EVAL_WORKERS,
    answer_urls: list[str] | None = None,
    protocol: str = "native",
    retest_singles: bool = False,
) -> dict:
    """Keep one hint per question id.

    - Single-candidate questions: keep as-is (already flipped once at label time).
    - Multi-candidate questions: re-test each unique hint ``repeats`` times; pick
      highest flip rate, tie-break shorter hint.
    """
    labels_file = Path(labels_file)
    out_file = Path(out_file)
    rows = load_jsonl(labels_file)
    by_id: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)

    pool = _AnswererPool(answer_urls or ANSWER_URLS, cfg.answer_model)
    scoring_log: list[dict] = []
    selected: list[dict] = []

    multi_jobs: list[tuple[int, dict[str, dict]]] = []
    for qid, group in by_id.items():
        cands = _group_candidates(group)
        if len(cands) == 1 and not retest_singles:
            row = next(iter(cands.values()))
            selected.append({
                **row,
                "dedup_score": 1.0,
                "dedup_repeats": 0,
                "dedup_n_candidates": 1,
                "dedup_method": "single_flip",
            })
            scoring_log.append({
                "id": qid, "hint_key": _hint_key(row["label_hint"]),
                "score": 1.0, "ems": [1], "repeats": 0, "selected": True,
                "reason": "only_candidate",
            })
        else:
            multi_jobs.append((qid, cands))

    def _job(qid: int, cands: dict[str, dict]) -> tuple[int, dict, list[dict]]:
        scored: list[tuple[float, int, str, dict, list[int]]] = []
        logs: list[dict] = []
        for key, row in cands.items():
            score, ems = _score_hint(pool, cfg, row, repeats=repeats, protocol=protocol)
            scored.append((score, len(row["label_hint"]), key, row, ems))
            logs.append({
                "id": qid, "hint_key": key, "score": score, "ems": ems,
                "repeats": repeats, "hint_len": len(row["label_hint"]),
            })
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        best_score, _, best_key, best_row, best_ems = scored[0]
        for log in logs:
            log["selected"] = log["hint_key"] == best_key
        out_row = {
            **best_row,
            "dedup_score": best_score,
            "dedup_repeats": repeats,
            "dedup_n_candidates": len(cands),
            "dedup_method": "repeat_flip_rate",
            "dedup_ems": best_ems,
        }
        return qid, out_row, logs

    if multi_jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_job, qid, cands) for qid, cands in multi_jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="dedup-retest"):
                _, out_row, logs = fut.result()
                selected.append(out_row)
                scoring_log.extend(logs)

    selected.sort(key=lambda r: r["id"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_file, selected)
    scoring_path = out_file.with_suffix(".scoring.jsonl")
    write_jsonl(scoring_path, scoring_log)

    n_multi = len(multi_jobs)
    n_retest_calls = sum(
        len(_group_candidates(by_id[qid])) * repeats for qid, _ in multi_jobs
    )
    stats = {
        "labels_in": len(rows),
        "labels_out": len(selected),
        "unique_questions": len(by_id),
        "single_candidate": len(by_id) - n_multi,
        "multi_candidate": n_multi,
        "repeats": repeats,
        "retest_singles": retest_singles,
        "oss_retest_calls": n_retest_calls,
        "labels_file": str(labels_file),
        "out_file": str(out_file),
        "scoring_file": str(scoring_path),
        "answer_urls": pool.urls,
        "protocol": protocol,
    }
    stats_path = out_file.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(
        f"dedup: {len(rows)} -> {len(selected)} labels ({len(by_id)} questions, "
        f"multi={n_multi}, oss_retests={n_retest_calls}) -> {out_file}"
    )
    return stats
