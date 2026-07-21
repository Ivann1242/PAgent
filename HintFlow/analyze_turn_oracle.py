#!/usr/bin/env python3
"""Analyze paired baseline/final/oracle performance for residual or legacy trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import (  # noqa: E402
    exact_match,
    extract_final_answer,
    has_parseable_answer,
    load_jsonl,
    write_jsonl,
)


def _bootstrap_ci(
    deltas: list[int],
    *,
    samples: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    draws = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(max(samples, 1))
    ]
    draws.sort()
    lo = draws[int((alpha / 2) * (len(draws) - 1))]
    hi = draws[int((1 - alpha / 2) * (len(draws) - 1))]
    return lo, hi


def _legacy_candidate_ems(record: dict[str, Any]) -> list[int]:
    gold = str(record.get("gold") or "")
    out: list[int] = []
    for step in record.get("steps") or []:
        pred = extract_final_answer(step.get("observation") or "")
        out.append(exact_match(pred, gold))
    return out


def relabel_residual_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        if record.get("error") or not record.get("candidates"):
            continue
        gold = str(record.get("gold") or "")
        for candidate in record.get("candidates") or []:
            solution = str(candidate.get("solution") or "")
            answer = extract_final_answer(solution)
            candidate["answer"] = answer
            candidate["parseable"] = has_parseable_answer(solution)
            candidate["em"] = exact_match(answer, gold) if gold else None
        incumbent_index = int(record.get("incumbent_index") or 0)
        candidates = record["candidates"]
        record["baseline_em"] = int(candidates[0].get("em") or 0)
        record["em"] = int(candidates[incumbent_index].get("em") or 0)
        record["final_answer"] = candidates[incumbent_index].get("answer") or ""
        record["oracle_em"] = max(int(c.get("em") or 0) for c in candidates)
    return records


def summarize_legacy(
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    baseline = {
        r["id"]: int(r.get("em") or 0) for r in (baseline_records or [])
    }
    final: list[int] = []
    oracle: list[int] = []
    base: list[int] = []
    for row in records:
        candidate_ems = _legacy_candidate_ems(row)
        final.append(int(row.get("em") or 0))
        oracle.append(max(candidate_ems, default=int(row.get("em") or 0)))
        if baseline:
            base.append(baseline.get(row.get("id"), 0))
    result: dict[str, Any] = {
        "schema": "legacy_hintflow",
        "n": len(records),
        "final_em": mean(final) if final else 0.0,
        "any_turn_oracle_em": mean(oracle) if oracle else 0.0,
        "lost_correct_before_final": sum(
            int(o and not f) for o, f in zip(oracle, final)
        ),
    }
    if base:
        result.update(
            {
                "baseline_em": mean(base),
                "union_baseline_any_turn_em": mean(
                    max(b, o) for b, o in zip(base, oracle)
                ),
            }
        )
    return result


def summarize_residual(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    good = [r for r in records if not r.get("error") and r.get("candidates")]
    # Primary metrics include failed requests as incorrect; successful-only
    # diagnostics are available through n_error and action stats.
    baseline = [
        int(r.get("baseline_em") or 0)
        if not r.get("error") and r.get("candidates")
        else 0
        for r in records
    ]
    final = [
        int(r.get("em") or 0)
        if not r.get("error") and r.get("candidates")
        else 0
        for r in records
    ]
    oracle = [
        int(
            r.get("oracle_em")
            if r.get("oracle_em") is not None
            else max(
                (int(c.get("em") or 0) for c in r.get("candidates") or []),
                default=0,
            )
        )
        if not r.get("error") and r.get("candidates")
        else 0
        for r in records
    ]
    deltas = [f - b for b, f in zip(baseline, final)]
    lo, hi = _bootstrap_ci(
        deltas, samples=bootstrap_samples, seed=seed
    )
    baseline_correct = sum(baseline)
    baseline_wrong = len(baseline) - baseline_correct
    recovered = sum(int(not b and f) for b, f in zip(baseline, final))
    harmed = sum(int(b and not f) for b, f in zip(baseline, final))
    oracle_recovered = sum(int(not b and o) for b, o in zip(baseline, oracle))

    action_stats: dict[str, dict[str, float | int]] = {}
    raw_action: dict[str, dict[str, list[int] | int]] = defaultdict(
        lambda: {
            "attempts": 0,
            "candidate_correct": 0,
            "local_improve": 0,
            "local_harm": 0,
            "selected": 0,
        }
    )
    selector_strict_n = selector_strict_correct = 0
    for row in good:
        candidates = row.get("candidates") or []
        for turn in row.get("turns") or []:
            ci = turn.get("candidate_index")
            if ci is None or ci >= len(candidates):
                continue
            before_i = int(turn.get("incumbent_before") or 0)
            after_i = int(turn.get("incumbent_after") or 0)
            challenger = candidates[ci]
            incumbent = candidates[before_i]
            action = str(turn.get("action") or challenger.get("action") or "UNKNOWN")
            c_em = int(challenger.get("em") or 0)
            i_em = int(incumbent.get("em") or 0)
            bucket = raw_action[action]
            bucket["attempts"] = int(bucket["attempts"]) + 1
            bucket["candidate_correct"] = int(bucket["candidate_correct"]) + c_em
            bucket["local_improve"] = int(bucket["local_improve"]) + int(c_em > i_em)
            bucket["local_harm"] = int(bucket["local_harm"]) + int(c_em < i_em)
            bucket["selected"] = int(bucket["selected"]) + int(after_i == ci)

            if c_em != i_em:
                selector_strict_n += 1
                should_replace = c_em > i_em
                did_replace = after_i == ci
                selector_strict_correct += int(should_replace == did_replace)

    for action, values in raw_action.items():
        attempts = int(values["attempts"])
        action_stats[action] = {
            **{k: int(v) for k, v in values.items()},
            "candidate_accuracy": (
                int(values["candidate_correct"]) / attempts if attempts else 0.0
            ),
            "local_recovery_rate": (
                int(values["local_improve"]) / attempts if attempts else 0.0
            ),
        }

    n = len(records)
    return {
        "schema": "residual",
        "n": n,
        "n_error": len(records) - len(good),
        "baseline_em": mean(baseline) if baseline else 0.0,
        "final_em": mean(final) if final else 0.0,
        "paired_delta": mean(deltas) if deltas else 0.0,
        "paired_delta_bootstrap_95ci": [lo, hi],
        "any_candidate_oracle_em": mean(oracle) if oracle else 0.0,
        "oracle_headroom_vs_baseline": (
            mean(o - b for b, o in zip(baseline, oracle)) if baseline else 0.0
        ),
        "selector_oracle_gap": (
            mean(o - f for f, o in zip(final, oracle)) if final else 0.0
        ),
        "baseline_correct": baseline_correct,
        "baseline_wrong": baseline_wrong,
        "recovered_wrong_to_right": recovered,
        "harmed_right_to_wrong": harmed,
        "net_flips": recovered - harmed,
        "baseline_correct_retention": (
            (baseline_correct - harmed) / baseline_correct
            if baseline_correct
            else 0.0
        ),
        "baseline_wrong_recovery_rate": (
            recovered / baseline_wrong if baseline_wrong else 0.0
        ),
        "oracle_recoverable_baseline_errors": oracle_recovered,
        "avg_solver_calls": (
            mean(len(r.get("candidates") or []) for r in records)
            if records
            else 0.0
        ),
        "selector_strict_accuracy": (
            selector_strict_correct / selector_strict_n
            if selector_strict_n
            else 0.0
        ),
        "selector_strict_n": selector_strict_n,
        "action_stats": action_stats,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="residual or legacy trajectory JSONL")
    p.add_argument("--baseline", default="", help="legacy baseline JSONL")
    p.add_argument("--out", default="", help="optional summary JSON path")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--relabel", action="store_true")
    p.add_argument("--relabel-out", default="")
    args = p.parse_args()

    records = load_jsonl(Path(args.input))
    is_residual = any(r.get("candidates") is not None for r in records)
    if is_residual:
        if args.relabel:
            records = relabel_residual_records(records)
            if args.relabel_out:
                write_jsonl(Path(args.relabel_out), records)
        summary = summarize_residual(
            records,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    else:
        baseline = load_jsonl(Path(args.baseline)) if args.baseline else None
        summary = summarize_legacy(records, baseline)
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
