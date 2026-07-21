#!/usr/bin/env python3
"""Export grouped multi-task supervision from residual counterfactual records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core import (  # noqa: E402
    exact_match,
    extract_final_answer,
    has_parseable_answer,
    hint_leaks_gold,
    load_dapo_rows,
    load_jsonl,
    write_jsonl,
)
from residual_agent import (  # noqa: E402
    ACTION_SYSTEM,
    CORRECTNESS_SYSTEM,
    DIAGNOSIS_SYSTEM,
    RESIDUAL_ACTIONS,
    SELECTOR_SYSTEM,
    build_action_prompt,
    build_correctness_prompt,
    build_diagnosis_prompt,
    build_selection_prompt,
)


def _split_for(problem_id: object, *, seed: int, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(f"{seed}:{problem_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def _problem_hash(problem: str) -> str:
    normalized = " ".join((problem or "").split()).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _safe_teacher_text(text: str, gold: str) -> str:
    value = str(text or "")
    if hint_leaks_gold(value, gold):
        return ""
    return re.sub(
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?",
        "<number>",
        value,
    )


def _row(
    *,
    problem_id: object,
    task: str,
    system: str,
    prompt: str,
    target: str,
    weight: float = 1.0,
    options: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "task": task,
        "system": system,
        "prompt": prompt,
        "target": target,
        "weight": float(weight),
        "options": options or [],
        "metadata": metadata or {},
    }


def _correctness_row(
    problem_id: object,
    problem: str,
    candidate: dict,
    *,
    source: str,
) -> dict:
    em = int(candidate.get("em") or 0)
    row = _row(
        problem_id=problem_id,
        task="correctness",
        system=CORRECTNESS_SYSTEM,
        prompt=build_correctness_prompt(problem, candidate.get("solution") or ""),
        target="CORRECT" if em else "INCORRECT",
        options=["CORRECT", "INCORRECT"],
        metadata={
            "source": source,
            "candidate_action": candidate.get("action"),
            "em": em,
        },
    )
    row["problem_hash"] = _problem_hash(problem)
    return row


def _selection_row(
    problem_id: object,
    problem: str,
    incumbent: dict,
    challenger: dict,
    *,
    source: str,
) -> dict | None:
    incumbent_em = int(incumbent.get("em") or 0)
    challenger_em = int(challenger.get("em") or 0)
    is_tie = incumbent_em == challenger_em
    target = "REPLACE" if challenger_em > incumbent_em else "KEEP"
    row = _row(
        problem_id=problem_id,
        task="selection",
        system=SELECTOR_SYSTEM,
        prompt=build_selection_prompt(
            problem,
            incumbent.get("solution") or "",
            challenger.get("solution") or "",
        ),
        target=target,
        weight=0.5 if is_tie else 2.0,
        options=["KEEP", "REPLACE"],
        metadata={
            "source": source,
            "incumbent_em": incumbent_em,
            "challenger_em": challenger_em,
            "challenger_action": challenger.get("action"),
            "tie": is_tie,
        },
    )
    row["problem_hash"] = _problem_hash(problem)
    return row


def _action_rows(record: dict, *, action_cost: float) -> list[dict]:
    values = record.get("action_values") or {}
    if not values:
        return []
    adjusted: dict[str, float] = {}
    for action in RESIDUAL_ACTIONS:
        if action not in values:
            continue
        q_value = float(values[action].get("q") or 0.0)
        adjusted[action] = q_value - (action_cost if action != "STOP" else 0.0)
    if not adjusted:
        return []
    stop_value = adjusted.get("STOP", 0.0)
    positive_actions = [
        action
        for action in RESIDUAL_ACTIONS
        if action != "STOP"
        and action in adjusted
        and adjusted[action] > stop_value
    ]
    targets = positive_actions or ["STOP"]
    sorted_values = sorted(adjusted.values(), reverse=True)
    margin = (
        sorted_values[0] - sorted_values[1] if len(sorted_values) > 1 else 0.0
    )
    feedback = record.get("policy_feedback") or {}
    incumbent = record.get("incumbent") or {}
    p_correct_raw = feedback.get("p_correct")
    remaining_raw = record.get("remaining_calls")
    prompt = build_action_prompt(
        record.get("problem") or "",
        incumbent.get("solution") or "",
        p_correct=(
            float(p_correct_raw) if p_correct_raw is not None else 0.5
        ),
        error_type=str(feedback.get("error_type") or "UNKNOWN"),
        evidence=str(feedback.get("evidence") or ""),
        tried_actions=[str(x) for x in record.get("tried_actions") or []],
        remaining_calls=int(remaining_raw) if remaining_raw is not None else 6,
    )
    rows = []
    for target in targets:
        advantage = max(adjusted[target] - stop_value, 0.0)
        row = _row(
            problem_id=record.get("problem_id", record.get("id")),
            task="action",
            system=ACTION_SYSTEM,
            prompt=prompt,
            target=target,
            weight=1.0 + advantage,
            options=[a for a in RESIDUAL_ACTIONS if a in adjusted],
            metadata={
                "source": "counterfactual_awr",
                "q_values": adjusted,
                "raw_q_values": {
                    a: float(values[a].get("q") or 0.0) for a in adjusted
                },
                "margin": margin,
                "advantage": advantage,
                "baseline_em": int(record.get("baseline_em") or 0),
            },
        )
        row["problem_hash"] = _problem_hash(record.get("problem") or "")
        rows.append(row)
    return rows


def _diagnosis_row(record: dict) -> dict | None:
    teacher = record.get("teacher_feedback") or {}
    incumbent = record.get("incumbent") or {}
    if not teacher or not incumbent:
        return None
    if int(record.get("baseline_em") or 0):
        error_type, evidence, repair_hint = "NONE", "", ""
    else:
        error_type = str(teacher.get("error_type") or "UNKNOWN").upper()
        if error_type == "NONE":
            error_type = "UNKNOWN"
        gold = str(record.get("gold") or "")
        evidence = _safe_teacher_text(
            str(teacher.get("evidence") or ""), gold
        )
        repair_hint = _safe_teacher_text(
            str(teacher.get("repair_hint") or ""), gold
        )
    target = json.dumps(
        {
            "error_type": error_type,
            "evidence": evidence,
            "repair_hint": repair_hint,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    row = _row(
        problem_id=record.get("problem_id", record.get("id")),
        task="diagnosis",
        system=DIAGNOSIS_SYSTEM,
        prompt=build_diagnosis_prompt(
            record.get("problem") or "", incumbent.get("solution") or ""
        ),
        target=target,
        weight=0.5,
        metadata={
            "source": "counterfactual_teacher",
            "baseline_em": int(record.get("baseline_em") or 0),
        },
    )
    row["problem_hash"] = _problem_hash(record.get("problem") or "")
    return row


def export_counterfactual(
    records: Iterable[dict],
    *,
    action_cost: float,
) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        if record.get("error") or not record.get("incumbent"):
            continue
        gold = str(record.get("gold") or "")
        incumbent = record["incumbent"]
        incumbent_answer = extract_final_answer(incumbent.get("solution") or "")
        incumbent["answer"] = incumbent_answer
        incumbent["em"] = exact_match(incumbent_answer, gold)
        by_action: dict[str, list[int]] = defaultdict(list)
        for branch in record.get("branches") or []:
            candidate = branch.get("candidate") or {}
            answer = extract_final_answer(candidate.get("solution") or "")
            candidate["answer"] = answer
            candidate["em"] = exact_match(answer, gold)
            by_action[str(branch.get("action") or candidate.get("action"))].append(
                int(candidate["em"])
            )
        incumbent_em = int(incumbent["em"])
        record["baseline_em"] = incumbent_em
        record["oracle_em"] = max(
            [incumbent_em] + [value for values in by_action.values() for value in values]
        )
        record["action_values"] = {
            "STOP": {
                "q": float(incumbent_em),
                "advantage": 0.0,
                "candidate_em_mean": float(incumbent_em),
                "n": 1,
            },
            **{
                action: {
                    "q": sum(max(incumbent_em, value) for value in values)
                    / len(values),
                    "advantage": (
                        sum(max(incumbent_em, value) for value in values)
                        / len(values)
                        - incumbent_em
                    ),
                    "candidate_em_mean": sum(values) / len(values),
                    "n": len(values),
                }
                for action, values in by_action.items()
                if values
            },
        }
        problem_id = record.get("problem_id", record.get("id"))
        problem = record.get("problem") or ""
        rows.append(
            _correctness_row(
                problem_id, problem, incumbent, source="counterfactual_incumbent"
            )
        )
        seen_solutions = {hashlib.sha1((incumbent.get("solution") or "").encode()).hexdigest()}
        for branch in record.get("branches") or []:
            candidate = branch.get("candidate") or {}
            solution_hash = hashlib.sha1(
                (candidate.get("solution") or "").encode()
            ).hexdigest()
            if solution_hash not in seen_solutions:
                rows.append(
                    _correctness_row(
                        problem_id,
                        problem,
                        candidate,
                        source="counterfactual_branch",
                    )
                )
                seen_solutions.add(solution_hash)
            selection = _selection_row(
                problem_id,
                problem,
                incumbent,
                candidate,
                source="counterfactual_branch",
            )
            if selection is not None:
                rows.append(selection)
        rows.extend(_action_rows(record, action_cost=action_cost))
        diagnosis = _diagnosis_row(record)
        if diagnosis is not None:
            rows.append(diagnosis)
    return rows


def _legacy_candidates_from_eval(path: Path) -> Iterable[tuple[object, str, list[dict]]]:
    for record in load_jsonl(path):
        problem_id = record.get("id")
        problem = record.get("problem") or ""
        gold = str(record.get("gold") or "")
        candidates = []
        seen = set()
        for index, step in enumerate(record.get("steps") or []):
            solution = step.get("observation") or ""
            digest = hashlib.sha1(solution.encode()).hexdigest()
            if (
                not solution
                or digest in seen
                or not has_parseable_answer(solution)
            ):
                continue
            seen.add(digest)
            answer = extract_final_answer(solution)
            candidates.append(
                {
                    "index": index,
                    "action": ((step.get("review") or {}).get("action")) or "LEGACY",
                    "solution": solution,
                    "answer": answer,
                    "em": exact_match(answer, gold),
                }
            )
        yield problem_id, problem, candidates


def _legacy_candidates_from_trees(path: Path) -> Iterable[tuple[object, str, list[dict]]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            tree = json.loads(line)
            problem_id = tree.get("id")
            problem = tree.get("problem") or ""
            gold = str(tree.get("gold") or "")
            candidates = []
            seen = set()
            for index, node in enumerate(tree.get("nodes") or []):
                solution = node.get("observation") or ""
                if (
                    node.get("kind") != "review"
                    or not solution
                    or not has_parseable_answer(solution)
                ):
                    continue
                digest = hashlib.sha1(solution.encode()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                answer = extract_final_answer(solution)
                parsed = node.get("orch_parsed") or {}
                candidates.append(
                    {
                        "index": index,
                        "action": parsed.get("action") or "LEGACY",
                        "solution": solution,
                        "answer": answer,
                        "em": exact_match(answer, gold),
                    }
                )
            yield problem_id, problem, candidates


def export_legacy(
    groups: Iterable[tuple[object, str, list[dict]]],
    *,
    max_selection_pairs: int,
    seed: int,
    tasks: set[str],
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for problem_id, problem, candidates in groups:
        if "correctness" in tasks:
            for candidate in candidates:
                rows.append(
                    _correctness_row(
                        problem_id, problem, candidate, source="legacy_trajectory"
                    )
                )
        if "selection" not in tasks:
            continue
        correct = [c for c in candidates if int(c.get("em") or 0)]
        wrong = [c for c in candidates if not int(c.get("em") or 0)]
        rng.shuffle(correct)
        rng.shuffle(wrong)
        n_pairs = min(max_selection_pairs, len(correct) * len(wrong))
        if not n_pairs:
            continue
        pairs = [(c, w) for c in correct for w in wrong]
        rng.shuffle(pairs)
        for i, (correct_candidate, wrong_candidate) in enumerate(pairs[:n_pairs]):
            if i % 2:
                incumbent, challenger = correct_candidate, wrong_candidate
            else:
                incumbent, challenger = wrong_candidate, correct_candidate
            selection = _selection_row(
                problem_id,
                problem,
                incumbent,
                challenger,
                source="legacy_trajectory",
            )
            if selection is not None:
                rows.append(selection)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--counterfactual",
        default=str(
            _ROOT
            / "checkpoints"
            / "residual_feedback_collection"
            / "turns.jsonl"
        ),
        help="comma-separated counterfactual JSONL files",
    )
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "checkpoints" / "residual_feedback_dataset"),
    )
    p.add_argument("--legacy-evals", default="")
    p.add_argument("--legacy-trees", default="")
    p.add_argument(
        "--exclude-data-files",
        default=str(_ROOT / "data" / "DAPO-Math.parquet"),
        help="comma-separated final-eval datasets whose problem hashes must be excluded",
    )
    p.add_argument(
        "--split-manifest",
        default=str(_ROOT / "checkpoints" / "residual_splits" / "manifest.json"),
        help="immutable problem-hash manifest; dev becomes val and final is excluded",
    )
    p.add_argument("--max-legacy-selection-pairs", type=int, default=4)
    p.add_argument(
        "--legacy-tasks",
        default="correctness,selection",
        help="legacy tree/eval tasks to reuse",
    )
    p.add_argument("--action-cost", type=float, default=0.02)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--allow-unknown-problems", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("invalid val/test ratios")

    records: list[dict] = []
    for value in [
        x.strip() for x in args.counterfactual.split(",") if x.strip()
    ]:
        path = Path(value)
        if path.exists():
            records.extend(load_jsonl(path))
    rows = export_counterfactual(records, action_cost=args.action_cost)
    legacy_tasks = {
        value.strip()
        for value in args.legacy_tasks.split(",")
        if value.strip()
    }
    for value in [x.strip() for x in args.legacy_evals.split(",") if x.strip()]:
        rows.extend(
            export_legacy(
                _legacy_candidates_from_eval(Path(value)),
                max_selection_pairs=args.max_legacy_selection_pairs,
                seed=args.seed,
                tasks=legacy_tasks,
            )
        )
    for value in [x.strip() for x in args.legacy_trees.split(",") if x.strip()]:
        rows.extend(
            export_legacy(
                _legacy_candidates_from_trees(Path(value)),
                max_selection_pairs=args.max_legacy_selection_pairs,
                seed=args.seed,
                tasks=legacy_tasks,
            )
        )

    excluded_hashes: set[str] = set()
    excluded_files: list[str] = []
    for value in [
        x.strip() for x in args.exclude_data_files.split(",") if x.strip()
    ]:
        path = Path(value)
        if not path.exists():
            continue
        excluded_files.append(str(path))
        source_rows = load_dapo_rows(path) if path.suffix == ".parquet" else load_jsonl(path)
        excluded_hashes.update(
            _problem_hash(str(source.get("problem") or "")) for source in source_rows
        )
    before_exclusion = len(rows)
    rows = [row for row in rows if row.get("problem_hash") not in excluded_hashes]

    manifest: dict[str, Any] = {}
    manifest_path = Path(args.split_manifest) if args.split_manifest else None
    if manifest_path:
        if not manifest_path.exists():
            raise SystemExit(f"split manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_hashes = set(
        ((manifest.get("splits") or {}).get("train") or {}).get("problem_hashes")
        or []
    )
    dev_hashes = set(
        ((manifest.get("splits") or {}).get("dev") or {}).get("problem_hashes")
        or []
    )
    final_hashes = set(
        ((manifest.get("splits") or {}).get("final") or {}).get("problem_hashes")
        or []
    )
    rows = [row for row in rows if row.get("problem_hash") not in final_hashes]
    unknown_hashes: set[str] = set()
    if manifest:
        allowed_hashes = train_hashes | dev_hashes
        unknown_hashes = {
            str(row.get("problem_hash"))
            for row in rows
            if row.get("problem_hash") not in allowed_hashes
        }
        if unknown_hashes and not args.allow_unknown_problems:
            raise SystemExit(
                f"{len(unknown_hashes)} problem hashes absent from split manifest"
            )
        rows = [
            row for row in rows if row.get("problem_hash") in allowed_hashes
        ]

    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        digest = row.get("problem_hash")
        if digest in dev_hashes:
            split = "val"
        else:
            split = _split_for(
                digest or row["problem_id"],
                seed=args.seed,
                val_ratio=0.0 if manifest else args.val_ratio,
                test_ratio=args.test_ratio,
            )
        by_split[split].append(row)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "counterfactual": args.counterfactual,
        "action_cost": args.action_cost,
        "seed": args.seed,
        "excluded_data_files": excluded_files,
        "n_excluded_rows": before_exclusion - len(rows),
        "n_unknown_problem_hashes": len(unknown_hashes),
        "split_manifest": str(manifest_path) if manifest else "",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_rows = by_split.get(split, [])
        write_jsonl(out_dir / f"{split}.jsonl", split_rows)
        summary["splits"][split] = {
            "n": len(split_rows),
            "by_task": dict(Counter(r["task"] for r in split_rows)),
            "n_problems": len({str(r.get("problem_hash")) for r in split_rows}),
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
