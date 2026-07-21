#!/usr/bin/env python3
"""Evaluate the baseline-first residual agent with an exact paired turn-0 baseline."""

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
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_turn_oracle import summarize_residual  # noqa: E402
from config import EVAL_PARQUET  # noqa: E402
from core import append_jsonl, load_dapo_rows, load_jsonl, write_jsonl  # noqa: E402
from residual_agent import (  # noqa: E402
    ORCH_MODEL,
    ORCH_URL,
    SOLVER_MODEL,
    ResidualHintFlowAgent,
)


class _RoundRobin:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            url = self.urls[self._i % len(self.urls)]
            self._i += 1
            return url


def _row_seed(base_seed: int, row_id: object) -> int:
    digest = hashlib.sha1(str(row_id).encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16) % 1_000_000


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _eval_row(
    row: dict,
    *,
    solver_url: str,
    orch_url: str,
    orch_model: str,
    solver_model: str,
    max_solver_calls: int,
    solver_max_tokens: int,
    branch_max_tokens: int,
    request_timeout: float,
    branch_temperature: float,
    policy_mode: str,
    selector_mode: str,
    feedback_mode: str,
    replace_threshold: float,
    seed: int,
) -> dict:
    try:
        agent = ResidualHintFlowAgent(
            orch_url=orch_url,
            orch_model=orch_model,
            solver_url=solver_url,
            solver_model=solver_model,
            max_solver_calls=max_solver_calls,
            solver_max_tokens=solver_max_tokens,
            branch_max_tokens=branch_max_tokens,
            request_timeout=request_timeout,
            branch_temperature=branch_temperature,
            policy_mode=policy_mode,
            selector_mode=selector_mode,
            feedback_mode=feedback_mode,
            replace_threshold=replace_threshold,
        )
        traj = agent.run(
            row["problem"],
            gold=str(row.get("gold") or ""),
            seed=_row_seed(seed, row.get("id")),
        )
        record = traj.to_dict()
        record.update(
            {
                "id": row.get("id"),
                "solver_url": solver_url,
                "error": None,
            }
        )
        return record
    except Exception as exc:
        return {
            "id": row.get("id"),
            "problem": row.get("problem"),
            "gold": str(row.get("gold") or ""),
            "candidates": [],
            "turns": [],
            "incumbent_index": 0,
            "final_answer": "",
            "baseline_em": 0,
            "em": 0,
            "oracle_em": 0,
            "solver_url": solver_url,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _load_rows(path: Path, limit: int, offset: int) -> list[dict]:
    if path.suffix == ".parquet":
        rows = load_dapo_rows(path)
    else:
        rows = load_jsonl(path)
    if offset:
        rows = rows[offset:]
    return rows[:limit] if limit > 0 else rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", default=str(EVAL_PARQUET))
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--solver-urls", default="http://127.0.0.1:8006/v1")
    p.add_argument("--orch-url", default=ORCH_URL)
    p.add_argument("--orch-model", default=ORCH_MODEL)
    p.add_argument("--solver-model", default=SOLVER_MODEL)
    p.add_argument("--model-tag", default="", help="checkpoint path/hash served at orch-url")
    p.add_argument("--max-solver-calls", type=int, default=7)
    p.add_argument("--solver-max-tokens", type=int, default=8192)
    p.add_argument("--branch-max-tokens", type=int, default=4096)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--branch-temperature", type=float, default=0.2)
    p.add_argument("--policy-mode", choices=("fixed", "adaptive"), default="fixed")
    p.add_argument(
        "--selector-mode", choices=("orch", "keep", "replace"), default="orch"
    )
    p.add_argument(
        "--feedback-mode", choices=("json", "trained"), default="json"
    )
    p.add_argument("--replace-threshold", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "checkpoints" / "eval_residual_128"),
    )
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    urls = [u.strip() for u in args.solver_urls.split(",") if u.strip()]
    if not urls:
        raise SystemExit("no solver urls")
    if not 1 <= args.max_solver_calls <= 7:
        raise SystemExit("max-solver-calls must be in [1, 7]")
    if args.workers < 1:
        raise SystemExit("workers must be >=1")
    rows = _load_rows(Path(args.data_file), args.limit, args.offset)
    if not rows:
        raise SystemExit(f"no rows in {args.data_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".eval.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another eval owns {lock_path}")
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()
    out_file = out_dir / "residual.jsonl"
    data_path = Path(args.data_file)
    data_stat = data_path.stat()
    run_config = {
        "data_file": str(data_path.resolve()),
        "data_size": data_stat.st_size,
        "data_mtime_ns": data_stat.st_mtime_ns,
        "limit": args.limit,
        "offset": args.offset,
        "solver_urls": urls,
        "orch_url": args.orch_url,
        "orch_model": args.orch_model,
        "solver_model": args.solver_model,
        "model_tag": args.model_tag,
        "max_solver_calls": args.max_solver_calls,
        "solver_max_tokens": args.solver_max_tokens,
        "branch_max_tokens": args.branch_max_tokens,
        "request_timeout": args.request_timeout,
        "branch_temperature": args.branch_temperature,
        "policy_mode": args.policy_mode,
        "selector_mode": args.selector_mode,
        "feedback_mode": args.feedback_mode,
        "replace_threshold": args.replace_threshold,
        "seed": args.seed,
    }
    config_hash = _config_hash(run_config)
    manifest_path = out_dir / "run_manifest.json"
    if args.resume and out_file.exists():
        if not manifest_path.exists():
            raise SystemExit("refusing resume without run_manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            raise SystemExit("resume config mismatch; use a new out-dir")
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
    if args.resume and out_file.exists():
        existing = {
            str(r.get("id")): r
            for r in load_jsonl(out_file)
            if r.get("id") is not None and not r.get("error")
        }
    else:
        out_file.unlink(missing_ok=True)
    pending = [r for r in rows if str(r.get("id")) not in existing]

    print(
        f"Residual eval: n={len(rows)} pending={len(pending)} workers={args.workers} "
        f"solver_calls<={args.max_solver_calls} policy={args.policy_mode} "
        f"selector={args.selector_mode} feedback={args.feedback_mode}",
        flush=True,
    )
    t0 = time.time()
    rr = _RoundRobin(urls)
    write_lock = threading.Lock()
    new_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, max(len(pending), 1))) as pool:
        futures = {
            pool.submit(
                _eval_row,
                row,
                solver_url=rr.next(),
                orch_url=args.orch_url,
                orch_model=args.orch_model,
                solver_model=args.solver_model,
                max_solver_calls=args.max_solver_calls,
                solver_max_tokens=args.solver_max_tokens,
                branch_max_tokens=args.branch_max_tokens,
                request_timeout=args.request_timeout,
                branch_temperature=args.branch_temperature,
                policy_mode=args.policy_mode,
                selector_mode=args.selector_mode,
                feedback_mode=args.feedback_mode,
                replace_threshold=args.replace_threshold,
                seed=args.seed,
            ): row
            for row in pending
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="residual"
        ):
            record = future.result()
            new_records.append(record)
            with write_lock:
                append_jsonl(out_file, record)

    merged = {**existing}
    for record in new_records:
        merged[str(record.get("id"))] = record
    wanted = {str(r.get("id")): i for i, r in enumerate(rows)}
    records = sorted(
        (r for key, r in merged.items() if key in wanted),
        key=lambda r: wanted[str(r.get("id"))],
    )
    write_jsonl(out_file, records)

    metrics = summarize_residual(records, seed=args.seed)
    summary = {
        "meta": {
            "data_file": str(args.data_file),
            "n": len(rows),
            "workers": args.workers,
            "solver_urls": urls,
            "orch_url": args.orch_url,
            "orch_model": args.orch_model,
            "solver_model": args.solver_model,
            "model_tag": args.model_tag,
            "max_solver_calls": args.max_solver_calls,
            "solver_max_tokens": args.solver_max_tokens,
            "branch_max_tokens": args.branch_max_tokens,
            "branch_temperature": args.branch_temperature,
            "policy_mode": args.policy_mode,
            "selector_mode": args.selector_mode,
            "feedback_mode": args.feedback_mode,
            "replace_threshold": args.replace_threshold,
            "seed": args.seed,
            "config_hash": config_hash,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        **metrics,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "summary.json").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {out_dir}", flush=True)
    lock_handle.close()
    lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
