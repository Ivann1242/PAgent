#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import append_jsonl, load_jsonl, write_jsonl  # noqa: E402
from HintFlow_draft.agent import DraftStepAgent  # noqa: E402


def _record_summary(records: list[dict], args, elapsed_sec: float) -> dict:
    valid = [record for record in records if not record.get("error")]
    denominator = max(len(records), 1)
    decisions: Counter[str] = Counter()
    decisions_by_step: dict[int, Counter[str]] = defaultdict(Counter)
    step_counts: Counter[int] = Counter()
    selector_failures = 0
    selector_failure_samples: list[dict[str, str | int]] = []
    premature_answers = 0
    total_steps = 0

    for record in valid:
        steps = record.get("steps") or []
        step_counts[len(steps)] += 1
        total_steps += len(steps)
        for step in steps:
            decision = str(step.get("decision") or "")
            decisions[decision] += 1
            decisions_by_step[int(step.get("step_index") or 0)][decision] += 1
            selector_ok = bool(step.get("selector_parse_ok", True))
            selector_failures += int(not selector_ok)
            if not selector_ok and len(selector_failure_samples) < 5:
                selector_failure_samples.append(
                    {
                        "id": record.get("id"),
                        "step_index": int(step.get("step_index") or 0),
                        "raw": str(step.get("selector_raw") or "")[:500],
                        "reason": str(step.get("reason") or "")[:300],
                    }
                )
            premature_answers += int(bool(step.get("premature_answer")))

    hinted = decisions["HINTED"]
    selected_total = decisions["HINTED"] + decisions["BASELINE"]
    per_step = {}
    for step_index, counts in sorted(decisions_by_step.items()):
        n = counts["HINTED"] + counts["BASELINE"]
        per_step[str(step_index)] = {
            "n": n,
            "baseline": counts["BASELINE"],
            "hinted": counts["HINTED"],
            "hinted_rate": hinted_rate(counts["HINTED"], n),
        }

    return {
        "meta": {
            "mode": args.mode,
            "data_file": args.data_file,
            "n": len(records),
            "n_success": len(valid),
            "requested_n": len(records),
            "workers": args.workers,
            "solver_urls": args.solver_urls_list,
            "small_url": args.small_url,
            "small_model": args.small_model,
            "planner_url": args.planner_url or args.small_url,
            "planner_model": args.planner_model or args.small_model,
            "hinter_url": args.hinter_url or args.small_url,
            "hinter_model": args.hinter_model or args.small_model,
            "selector_url": args.selector_url or args.small_url,
            "selector_model": args.selector_model or args.small_model,
            "solver_model": args.solver_model,
            "max_steps": args.max_steps,
            "draft_max_tokens": args.draft_max_tokens,
            "step_max_tokens": args.step_max_tokens,
            "planner_max_tokens": args.planner_max_tokens,
            "hint_max_tokens": args.hint_max_tokens,
            "selector_max_tokens": args.selector_max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "elapsed_sec": round(elapsed_sec, 1),
        },
        args.mode: {
            "em": sum(int(record.get("em") or 0) for record in records)
            / denominator,
            "draft_em": sum(int(record.get("draft_em") or 0) for record in records)
            / denominator,
            "delta_vs_draft": (
                sum(int(record.get("em") or 0) for record in records)
                - sum(int(record.get("draft_em") or 0) for record in records)
            )
            / denominator,
            "recovered_vs_draft": sum(
                int(record.get("recovered") or 0) for record in valid
            ),
            "harmed_vs_draft": sum(
                int(record.get("harmed") or 0) for record in valid
            ),
            "final_parseable_rate": sum(
                int(bool(record.get("final_parseable"))) for record in valid
            )
            / max(len(valid), 1),
            "plan_parse_failures": sum(
                int(not record.get("plan_parse_ok", False)) for record in valid
            ),
            "plan_fallbacks": sum(
                int(bool(record.get("plan_fallback"))) for record in valid
            ),
            "plan_sanitized": sum(
                int(bool(record.get("plan_sanitized"))) for record in valid
            ),
            "step_count_distribution": {
                str(k): v for k, v in sorted(step_counts.items())
            },
            "mean_steps": total_steps / max(len(valid), 1),
            "decisions": dict(decisions),
            "hinted_rate": hinted_rate(hinted, selected_total),
            "decisions_by_step": per_step,
            "selector_parse_failures": selector_failures,
            "selector_parse_failure_samples": selector_failure_samples,
            "premature_selected_answers": premature_answers,
            "draft_votes": sum(
                int(bool(step.get("draft_vote")))
                for record in valid
                for step in record.get("steps") or []
            ),
            "mean_large_calls": sum(
                int(record.get("n_large_calls") or 0) for record in valid
            )
            / max(len(valid), 1),
            "mean_small_calls": sum(
                int(record.get("n_small_calls") or 0) for record in valid
            )
            / max(len(valid), 1),
            "n_error": sum(int(bool(record.get("error"))) for record in records),
            "sample_error": next(
                (record.get("error") for record in records if record.get("error")),
                None,
            ),
        },
    }


def hinted_rate(hinted: int, total: int) -> float:
    return hinted / total if total else 0.0


def _write_paired_summary(out_dir: Path) -> None:
    full_path = out_dir / "full.jsonl"
    baseline_path = out_dir / "baseline.jsonl"
    if not full_path.exists() or not baseline_path.exists():
        return
    full = {
        record["id"]: record
        for record in load_jsonl(full_path)
        if record.get("id") is not None and not record.get("error")
    }
    baseline = {
        record["id"]: record
        for record in load_jsonl(baseline_path)
        if record.get("id") is not None and not record.get("error")
    }
    ids = sorted(set(full) & set(baseline))
    n = len(ids)
    full_correct = sum(int(full[i].get("em") or 0) for i in ids)
    baseline_correct = sum(int(baseline[i].get("em") or 0) for i in ids)
    summary = {
        "n_paired": n,
        "full_em": full_correct / n if n else 0.0,
        "baseline_em": baseline_correct / n if n else 0.0,
        "paired_delta": (full_correct - baseline_correct) / n if n else 0.0,
        "recovered_vs_baseline": sum(
            int(not baseline[i].get("em") and bool(full[i].get("em"))) for i in ids
        ),
        "harmed_vs_baseline": sum(
            int(bool(baseline[i].get("em")) and not full[i].get("em")) for i in ids
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _error_record(row: dict, solver_url: str, mode: str, exc: Exception) -> dict:
    return {
        "id": row.get("id"),
        "problem": row.get("problem", ""),
        "gold": row.get("gold", ""),
        "mode": mode,
        "draft": "",
        "draft_em": 0,
        "plan": [],
        "steps": [],
        "final_answer": "",
        "final_parseable": False,
        "em": 0,
        "recovered": 0,
        "harmed": 0,
        "n_large_calls": 0,
        "n_small_calls": 0,
        "solver_url": solver_url,
        "error": f"{type(exc).__name__}: {exc}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic Draft-Plan-Select reasoning."
    )
    parser.add_argument("--mode", choices=("full", "baseline"), required=True)
    parser.add_argument("--data-file", default="data/dapo_holdout_512.jsonl")
    parser.add_argument("--out-dir", default="checkpoints/eval_hintflow_draft_holdout512")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--solver-urls",
        default="http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1",
    )
    parser.add_argument("--solver-model", default="qwen3-14b")
    parser.add_argument("--small-url", default="http://127.0.0.1:8086/v1")
    parser.add_argument(
        "--small-model",
        default="qwen3-4b-blind-ff-grpo-clean-iid-vp-step200",
    )
    parser.add_argument("--planner-url", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--hinter-url", default=None)
    parser.add_argument("--hinter-model", default=None)
    parser.add_argument("--selector-url", default=None)
    parser.add_argument("--selector-model", default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--draft-max-tokens", type=int, default=8192)
    parser.add_argument("--step-max-tokens", type=int, default=2048)
    parser.add_argument("--planner-max-tokens", type=int, default=768)
    parser.add_argument("--hint-max-tokens", type=int, default=256)
    parser.add_argument("--selector-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.data_file))
    by_id = {}
    for row in rows:
        if row.get("id") not in by_id:
            by_id[row["id"]] = row
    rows = [by_id[key] for key in sorted(by_id)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {args.data_file}")

    args.solver_urls_list = [
        value.strip() for value in args.solver_urls.split(",") if value.strip()
    ]
    if not args.solver_urls_list:
        raise SystemExit("--solver-urls must contain at least one URL")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{args.mode}.jsonl"
    existing = {
        record["id"]: record
        for record in load_jsonl(out_jsonl)
        if record.get("id") is not None
    } if out_jsonl.exists() else {}
    done_ids = {
        record_id for record_id, record in existing.items() if not record.get("error")
    }
    todo = [row for row in rows if row["id"] not in done_ids]
    lock = threading.Lock()

    def work(row: dict) -> dict:
        numeric_id = int(row["id"])
        solver_url = args.solver_urls_list[numeric_id % len(args.solver_urls_list)]
        try:
            agent = DraftStepAgent(
                small_url=args.small_url,
                small_model=args.small_model,
                solver_url=solver_url,
                solver_model=args.solver_model,
                planner_url=args.planner_url,
                planner_model=args.planner_model,
                hinter_url=args.hinter_url,
                hinter_model=args.hinter_model,
                selector_url=args.selector_url,
                selector_model=args.selector_model,
                max_steps=args.max_steps,
                draft_max_tokens=args.draft_max_tokens,
                step_max_tokens=args.step_max_tokens,
                planner_max_tokens=args.planner_max_tokens,
                hint_max_tokens=args.hint_max_tokens,
                selector_max_tokens=args.selector_max_tokens,
                temperature=args.temperature,
                seed=args.seed + numeric_id * 100,
                request_timeout=args.request_timeout,
            )
            record = agent.run(
                row["problem"],
                gold=str(row.get("gold") or ""),
                mode=args.mode,
            ).to_dict()
            record.update({"id": row["id"], "solver_url": solver_url, "error": None})
            return record
        except Exception as exc:  # noqa: BLE001
            return _error_record(row, solver_url, args.mode, exc)

    print(
        f"draft eval mode={args.mode} n={len(rows)} todo={len(todo)} "
        f"workers={args.workers} small={args.small_model} solver={args.solver_model}",
        flush=True,
    )
    started = time.monotonic()
    if todo:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(todo))) as pool:
            futures = {pool.submit(work, row): row["id"] for row in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc=args.mode):
                record = future.result()
                existing[record["id"]] = record
                if not record.get("error"):
                    with lock:
                        append_jsonl(out_jsonl, record)
                else:
                    print(f"id={record['id']} {record['error']}", flush=True)

    records = [existing[row["id"]] for row in rows if row["id"] in existing]
    write_jsonl(out_jsonl, records)
    summary = _record_summary(records, args, time.monotonic() - started)
    summary_path = out_dir / f"{args.mode}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_paired_summary(out_dir)
    print(json.dumps(summary, indent=2))

    successful = sum(not record.get("error") for record in records)
    if successful != len(rows):
        raise SystemExit("evaluation incomplete; rerun to retry failed rows")


if __name__ == "__main__":
    main()
