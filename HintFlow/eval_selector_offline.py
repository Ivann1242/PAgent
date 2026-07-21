#!/usr/bin/env python3
"""Evaluate a trained selector on a frozen residual candidate archive."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for path in (_ROOT, _HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_turn_oracle import relabel_residual_records, summarize_residual  # noqa: E402
from core import load_jsonl, write_jsonl  # noqa: E402
from residual_agent import Candidate, ResidualHintFlowAgent  # noqa: E402


def _candidate(data: dict) -> Candidate:
    return Candidate(
        index=int(data.get("index") or 0),
        action=str(data.get("action") or ""),
        solution=str(data.get("solution") or ""),
        answer=str(data.get("answer") or ""),
        parseable=bool(data.get("parseable")),
        prompt=str(data.get("prompt") or ""),
        seed=data.get("seed"),
        em=data.get("em"),
    )


def _select_record(
    record: dict,
    *,
    orch_url: str,
    orch_model: str,
    threshold: float,
) -> dict:
    row = copy.deepcopy(record)
    try:
        agent = ResidualHintFlowAgent(
            orch_url=orch_url,
            orch_model=orch_model,
            solver_url="http://127.0.0.1:1/v1",
            max_solver_calls=1,
            feedback_mode="trained",
            selector_mode="orch",
            replace_threshold=threshold,
        )
        candidates = [_candidate(value) for value in row.get("candidates") or []]
        incumbent_index = 0
        selections = []
        for challenger_index in range(1, len(candidates)):
            selection = agent.select(
                str(row.get("problem") or ""),
                candidates[incumbent_index],
                candidates[challenger_index],
            )
            before = incumbent_index
            if (
                selection.decision == "REPLACE"
                and selection.confidence >= threshold
            ):
                incumbent_index = challenger_index
            selections.append(
                {
                    "challenger_index": challenger_index,
                    "incumbent_before": before,
                    "incumbent_after": incumbent_index,
                    "selection": asdict(selection),
                }
            )
        row["incumbent_index"] = incumbent_index
        row["offline_selections"] = selections
        row["error"] = None
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--orch-url", default="http://127.0.0.1:8086/v1")
    p.add_argument("--orch-model", default="qwen3-4b")
    p.add_argument("--model-tag", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--replace-threshold", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    all_records = relabel_residual_records(load_jsonl(Path(args.input)))
    records = [
        row
        for row in all_records
        if not row.get("error") and row.get("candidates")
    ]
    out_records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _select_record,
                record,
                orch_url=args.orch_url,
                orch_model=args.orch_model,
                threshold=args.replace_threshold,
            ): record
            for record in records
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="offline-selector"
        ):
            out_records.append(future.result())
    order = {str(row.get("id")): index for index, row in enumerate(records)}
    out_records.sort(key=lambda row: order[str(row.get("id"))])
    out_records = relabel_residual_records(out_records)
    summary = summarize_residual(out_records, seed=args.seed)
    summary["meta"] = {
        "input": args.input,
        "input_n": len(all_records),
        "skipped_input_errors": len(all_records) - len(records),
        "orch_url": args.orch_url,
        "orch_model": args.orch_model,
        "model_tag": args.model_tag,
        "replace_threshold": args.replace_threshold,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "residual.jsonl", out_records)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
