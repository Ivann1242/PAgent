#!/usr/bin/env python3
"""Collect one-step matched counterfactual data for residual turn feedback."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core import append_jsonl, hint_leaks_gold, load_dapo_rows, load_jsonl, write_jsonl  # noqa: E402
from residual_agent import (  # noqa: E402
    Candidate,
    NON_STOP_ACTIONS,
    ORCH_MODEL,
    ORCH_URL,
    SOLVER_MODEL,
    ResidualHintFlowAgent,
)


class _RoundRobin:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            url = self.urls[self._index % len(self.urls)]
            self._index += 1
            return url


def _stable_seed(base: int, row_id: object) -> int:
    digest = hashlib.sha1(f"{base}:{row_id}".encode("utf-8")).hexdigest()
    return base + int(digest[:8], 16) % 1_000_000


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_rows(path: Path) -> list[dict]:
    return load_dapo_rows(path) if path.suffix == ".parquet" else load_jsonl(path)


def _hashed_subset(rows: list[dict], size: int, seed: int) -> list[dict]:
    if size <= 0 or size >= len(rows):
        return rows
    return sorted(
        rows,
        key=lambda r: hashlib.sha1(
            f"{seed}:{r.get('id')}".encode("utf-8")
        ).hexdigest(),
    )[:size]


def _states_from_trajectories(path: Path) -> list[dict]:
    states: list[dict] = []
    for trajectory in load_jsonl(path):
        if trajectory.get("error"):
            continue
        candidates = trajectory.get("candidates") or []
        tried_actions: list[str] = []
        for turn_index, turn in enumerate(trajectory.get("turns") or []):
            incumbent_index = int(turn.get("incumbent_before") or 0)
            if incumbent_index >= len(candidates):
                if turn.get("action") and turn.get("action") != "STOP":
                    tried_actions.append(str(turn["action"]))
                continue
            states.append(
                {
                    "id": f"{trajectory.get('id')}:turn{turn_index}",
                    "problem_id": trajectory.get("id"),
                    "problem": trajectory.get("problem") or "",
                    "gold": str(trajectory.get("gold") or ""),
                    "_incumbent": candidates[incumbent_index],
                    "_turn_index": turn_index,
                    "_tried_actions": list(tried_actions),
                    "_remaining_calls": max(7 - (1 + turn_index), 0),
                }
            )
            if turn.get("action") and turn.get("action") != "STOP":
                tried_actions.append(str(turn["action"]))
    return states


def _collect_row(
    row: dict,
    *,
    solver_url: str,
    orch_url: str,
    orch_model: str,
    solver_model: str,
    actions: tuple[str, ...],
    samples_per_action: int,
    branch_temperature: float,
    solver_max_tokens: int,
    branch_max_tokens: int,
    request_timeout: float,
    branch_only_if_baseline_wrong: bool,
    seed: int,
) -> dict:
    problem = str(row.get("problem") or "")
    gold = str(row.get("gold") or "")
    row_seed = _stable_seed(seed, row.get("id"))
    try:
        agent = ResidualHintFlowAgent(
            orch_url=orch_url,
            orch_model=orch_model,
            solver_url=solver_url,
            solver_model=solver_model,
            max_solver_calls=7,
            solver_max_tokens=solver_max_tokens,
            branch_max_tokens=branch_max_tokens,
            request_timeout=request_timeout,
            branch_temperature=branch_temperature,
            policy_mode="fixed",
            selector_mode="keep",
            feedback_mode="json",
        )
        if row.get("_incumbent"):
            raw_incumbent = row["_incumbent"]
            incumbent = Candidate(
                index=0,
                action=str(raw_incumbent.get("action") or "INCUMBENT"),
                solution=str(raw_incumbent.get("solution") or ""),
                answer=str(raw_incumbent.get("answer") or ""),
                parseable=bool(raw_incumbent.get("parseable")),
                prompt=str(raw_incumbent.get("prompt") or ""),
                seed=raw_incumbent.get("seed"),
                em=int(raw_incumbent.get("em") or 0),
            )
        else:
            incumbent = agent.generate_baseline(
                problem, gold=gold, seed=row_seed, index=0
            )
        tried_actions = [str(x) for x in row.get("_tried_actions") or []]
        remaining_calls = int(
            row.get("_remaining_calls")
            if row.get("_remaining_calls") is not None
            else len(actions) * samples_per_action
        )
        policy_feedback = agent.feedback(
            problem,
            incumbent,
            tried_actions=tried_actions,
            remaining_calls=remaining_calls,
        )
        teacher = agent.teacher_feedback(problem, incumbent, gold)
        if hint_leaks_gold(teacher.repair_hint, gold):
            teacher.repair_hint = ""

        branches: list[dict] = []
        action_values: dict[str, dict] = {}
        next_index = 1
        active_actions = (
            ()
            if branch_only_if_baseline_wrong and int(incumbent.em or 0)
            else actions
        )
        for action in active_actions:
            candidates = []
            for sample_index in range(samples_per_action):
                candidate = agent.generate_action_candidate(
                    problem,
                    incumbent,
                    policy_feedback,
                    action,
                    variant=sample_index,
                    gold=gold,
                    # Common random seed across actions for each matched sample.
                    seed=row_seed + 1 + sample_index,
                    index=next_index,
                )
                next_index += 1
                candidate_dict = asdict(candidate)
                candidates.append(candidate_dict)
                branches.append(
                    {
                        "action": action,
                        "sample_index": sample_index,
                        "candidate": candidate_dict,
                    }
                )
            retained = [
                max(int(incumbent.em or 0), int(c["em"] or 0))
                for c in candidates
            ]
            q_value = mean(retained) if retained else float(incumbent.em or 0)
            action_values[action] = {
                "q": q_value,
                "advantage": q_value - float(incumbent.em or 0),
                "candidate_em_mean": (
                    mean(int(c["em"] or 0) for c in candidates)
                    if candidates
                    else 0.0
                ),
                "n": len(candidates),
            }

        stop_q = float(incumbent.em or 0)
        oracle_em = max(
            [int(incumbent.em or 0)]
            + [
                int(branch["candidate"].get("em") or 0)
                for branch in branches
            ]
        )
        return {
            "id": row.get("id"),
            "state_id": row.get("id"),
            "problem_id": row.get("problem_id", row.get("id")),
            "turn_index": row.get("_turn_index", 0),
            "problem": problem,
            "gold": gold,
            "solver_url": solver_url,
            "incumbent": asdict(incumbent),
            "policy_feedback": asdict(policy_feedback),
            "teacher_feedback": asdict(teacher),
            "branches": branches,
            "action_values": {
                "STOP": {
                    "q": stop_q,
                    "advantage": 0.0,
                    "candidate_em_mean": stop_q,
                    "n": 1,
                },
                **action_values,
            },
            "remaining_calls": remaining_calls,
            "tried_actions": tried_actions,
            "baseline_em": int(incumbent.em or 0),
            "oracle_em": oracle_em,
            "error": None,
        }
    except Exception as exc:
        return {
            "id": row.get("id"),
            "state_id": row.get("id"),
            "problem_id": row.get("problem_id", row.get("id")),
            "problem": problem,
            "gold": gold,
            "solver_url": solver_url,
            "incumbent": None,
            "branches": [],
            "action_values": {},
            "baseline_em": 0,
            "oracle_em": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", default=str(_ROOT / "data" / "train.jsonl"))
    p.add_argument(
        "--trajectory-source",
        default="",
        help="optional residual JSONL; branch every distinct visited incumbent state",
    )
    p.add_argument(
        "--out",
        default=str(
            _ROOT
            / "checkpoints"
            / "residual_feedback_collection"
            / "turns.jsonl"
        ),
    )
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--hash-subset",
        type=int,
        default=0,
        help="select this many rows by stable problem-id hash before offset/limit",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--solver-urls", default="http://127.0.0.1:8006/v1")
    p.add_argument("--orch-url", default=ORCH_URL)
    p.add_argument("--orch-model", default=ORCH_MODEL)
    p.add_argument("--solver-model", default=SOLVER_MODEL)
    p.add_argument(
        "--actions",
        default=",".join(NON_STOP_ACTIONS),
        help="comma-separated residual actions",
    )
    p.add_argument("--samples-per-action", type=int, default=2)
    p.add_argument("--branch-temperature", type=float, default=0.2)
    p.add_argument("--solver-max-tokens", type=int, default=8192)
    p.add_argument("--branch-max-tokens", type=int, default=4096)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument(
        "--branch-only-if-baseline-wrong",
        action="store_true",
        help="use gold only for collection allocation: correct incumbents produce STOP labels",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    actions = tuple(a.strip().upper() for a in args.actions.split(",") if a.strip())
    invalid = [a for a in actions if a not in NON_STOP_ACTIONS]
    if invalid:
        raise SystemExit(f"invalid actions: {invalid}")
    if args.samples_per_action < 1:
        raise SystemExit("samples-per-action must be >=1")
    if 1 + len(actions) * args.samples_per_action > 7:
        raise SystemExit(
            "counterfactual collection exceeds the hard 7-call OSS budget"
        )
    if args.workers < 1:
        raise SystemExit("workers must be >=1")
    urls = [u.strip() for u in args.solver_urls.split(",") if u.strip()]
    if not urls:
        raise SystemExit("no solver urls")

    rows = (
        _states_from_trajectories(Path(args.trajectory_source))
        if args.trajectory_source
        else _load_rows(Path(args.data_file))
    )
    rows = _hashed_subset(rows, args.hash_subset, args.seed)
    rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out.with_suffix(out.suffix + ".lock")
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another collector owns {lock_path}")
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()
    data_path = Path(args.trajectory_source or args.data_file)
    data_stat = data_path.stat()
    run_config = {
        "data_file": str(data_path.resolve()),
        "trajectory_source": args.trajectory_source,
        "data_size": data_stat.st_size,
        "data_mtime_ns": data_stat.st_mtime_ns,
        "limit": args.limit,
        "offset": args.offset,
        "hash_subset": args.hash_subset,
        "solver_urls": urls,
        "orch_url": args.orch_url,
        "orch_model": args.orch_model,
        "solver_model": args.solver_model,
        "actions": actions,
        "samples_per_action": args.samples_per_action,
        "branch_temperature": args.branch_temperature,
        "solver_max_tokens": args.solver_max_tokens,
        "branch_max_tokens": args.branch_max_tokens,
        "request_timeout": args.request_timeout,
        "branch_only_if_baseline_wrong": args.branch_only_if_baseline_wrong,
        "seed": args.seed,
    }
    config_hash = _config_hash(run_config)
    manifest_path = out.with_suffix(".manifest.json")
    if args.resume and out.exists():
        if not manifest_path.exists():
            raise SystemExit("refusing resume without matching manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            raise SystemExit("resume config mismatch; use a new output path")
    manifest_path.write_text(
        json.dumps(
            {"config_hash": config_hash, "config": run_config},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    existing: dict[str, dict] = {}
    if args.resume and out.exists():
        existing = {
            str(r.get("id")): r
            for r in load_jsonl(out)
            if r.get("id") is not None and not r.get("error")
        }
    else:
        out.unlink(missing_ok=True)
    pending = [r for r in rows if str(r.get("id")) not in existing]
    print(
        f"Residual feedback collection: n={len(rows)} pending={len(pending)} "
        f"workers={args.workers} actions={actions} samples/action={args.samples_per_action}",
        flush=True,
    )

    rr = _RoundRobin(urls)
    lock = threading.Lock()
    new_records: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(args.workers, max(len(pending), 1))) as pool:
        futures = {
            pool.submit(
                _collect_row,
                row,
                solver_url=rr.next(),
                orch_url=args.orch_url,
                orch_model=args.orch_model,
                solver_model=args.solver_model,
                actions=actions,
                samples_per_action=args.samples_per_action,
                branch_temperature=args.branch_temperature,
                solver_max_tokens=args.solver_max_tokens,
                branch_max_tokens=args.branch_max_tokens,
                request_timeout=args.request_timeout,
                branch_only_if_baseline_wrong=args.branch_only_if_baseline_wrong,
                seed=args.seed,
            ): row
            for row in pending
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="counterfactual"
        ):
            record = future.result()
            new_records.append(record)
            with lock:
                append_jsonl(out, record)

    merged = dict(existing)
    for record in new_records:
        merged[str(record.get("id"))] = record
    order = {str(r.get("id")): i for i, r in enumerate(rows)}
    records = sorted(
        (r for key, r in merged.items() if key in order),
        key=lambda r: order[str(r.get("id"))],
    )
    write_jsonl(out, records)
    good = [r for r in records if not r.get("error")]
    summary = {
        "data_file": args.data_file,
        "out": str(out),
        "n": len(records),
        "n_error": len(records) - len(good),
        "error_rate": (
            (len(records) - len(good)) / len(records) if records else 0.0
        ),
        "baseline_em": (
            mean(int(r.get("baseline_em") or 0) for r in records)
            if records
            else 0.0
        ),
        "oracle_em": (
            mean(int(r.get("oracle_em") or 0) for r in records)
            if records
            else 0.0
        ),
        "oracle_headroom": (
            mean(
                int(r.get("oracle_em") or 0)
                - int(r.get("baseline_em") or 0)
                for r in records
            )
            if records
            else 0.0
        ),
        "actions": list(actions),
        "samples_per_action": args.samples_per_action,
        "branch_temperature": args.branch_temperature,
        "config_hash": config_hash,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    lock_handle.close()
    lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
