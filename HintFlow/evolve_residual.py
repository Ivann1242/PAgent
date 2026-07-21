#!/usr/bin/env python3
"""Run/gate residual solve-observe-train-evaluate cycles with rollback-by-default."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def _read_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run(label: str, command: str) -> None:
    if not command:
        return
    print(f"=== {label} ===\n{command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def _metric(data: dict, *path: str, default: float = 0.0) -> float:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def gate(args: argparse.Namespace) -> dict[str, Any]:
    candidate = _read_json(args.candidate_summary)
    feedback = _read_json(args.feedback_meta)
    evaluation = _read_json(args.eval_summary)
    checks: list[dict[str, Any]] = []

    required = {
        "candidate_summary_present": bool(candidate),
        "feedback_meta_present": bool(feedback),
        "eval_summary_present": bool(evaluation),
    }
    required_names = (
        ("candidate_summary_present",)
        if args.stage == "candidate"
        else tuple(required)
    )
    for name in required_names:
        checks.append(
            {
                "name": name,
                "value": int(required[name]),
                "threshold": 1,
                "passed": required[name],
            }
        )
    if args.stage == "full":
        model_path = Path(args.model) if args.model else None
        model_valid = bool(
            model_path
            and model_path.is_dir()
            and (model_path / "config.json").exists()
        )
        checks.append(
            {
                "name": "candidate_model_valid",
                "value": int(model_valid),
                "threshold": 1,
                "passed": model_valid,
            }
        )
        eval_model_tag = str((evaluation.get("meta") or {}).get("model_tag") or "")
        checks.append(
            {
                "name": "eval_model_tag_matches",
                "value": eval_model_tag,
                "threshold": args.model,
                "passed": bool(eval_model_tag and eval_model_tag == args.model),
            }
        )

    if candidate:
        headroom = _metric(candidate, "oracle_headroom_vs_baseline")
        if not headroom:
            headroom = _metric(candidate, "oracle_headroom")
        checks.append(
            {
                "name": "candidate_oracle_headroom",
                "value": headroom,
                "threshold": args.min_oracle_headroom,
                "passed": headroom >= args.min_oracle_headroom,
            }
        )

    if feedback:
        history = feedback.get("history") or []
        latest = history[-1] if history else {}
        offline_selector_acc = _metric(
            latest, "metrics", "selection", "strict_accuracy"
        )
        online_selector_acc = _metric(
            evaluation, "selector_strict_accuracy", default=-1.0
        )
        selector_acc = (
            online_selector_acc
            if online_selector_acc >= 0.0
            else offline_selector_acc
        )
        correctness_acc = _metric(
            latest, "metrics", "correctness", "balanced_accuracy"
        )
        action_regret = _metric(
            latest, "metrics", "action", "mean_regret", default=1.0
        )
        generated = ((latest.get("metrics") or {}).get("free_generation") or {})
        generation_valid = min(
            (
                float(metrics.get("valid_rate") or 0.0)
                for task, metrics in generated.items()
                if task in {"action", "correctness", "diagnosis"}
            ),
            default=0.0,
        )
        checks.extend(
            [
                {
                    "name": "selector_accuracy",
                    "value": selector_acc,
                    "threshold": args.min_selector_accuracy,
                    "passed": selector_acc >= args.min_selector_accuracy,
                },
                {
                    "name": "correctness_accuracy",
                    "value": correctness_acc,
                    "threshold": args.min_correctness_accuracy,
                    "passed": correctness_acc >= args.min_correctness_accuracy,
                },
                {
                    "name": "action_regret",
                    "value": action_regret,
                    "threshold": args.max_action_regret,
                    "passed": action_regret <= args.max_action_regret,
                },
                {
                    "name": "free_generation_valid_rate",
                    "value": generation_valid,
                    "threshold": args.min_generation_valid,
                    "passed": generation_valid >= args.min_generation_valid,
                },
            ]
        )

    if evaluation:
        retention = _metric(evaluation, "baseline_correct_retention")
        delta = _metric(evaluation, "paired_delta")
        ci = evaluation.get("paired_delta_bootstrap_95ci") or [0.0, 0.0]
        ci_lower = float(ci[0]) if ci else 0.0
        checks.extend(
            [
                {
                    "name": "eval_error_rate",
                    "value": (
                        _metric(evaluation, "n_error")
                        / max(_metric(evaluation, "n"), 1.0)
                    ),
                    "threshold": args.max_error_rate,
                    "passed": (
                        _metric(evaluation, "n_error")
                        / max(_metric(evaluation, "n"), 1.0)
                    )
                    <= args.max_error_rate,
                },
                {
                    "name": "baseline_correct_retention",
                    "value": retention,
                    "threshold": args.min_retention,
                    "passed": retention >= args.min_retention,
                },
                {
                    "name": "paired_delta",
                    "value": delta,
                    "threshold": args.min_paired_delta,
                    "passed": delta >= args.min_paired_delta,
                },
                {
                    "name": "paired_ci_lower",
                    "value": ci_lower,
                    "threshold": args.min_ci_lower,
                    "passed": ci_lower > args.min_ci_lower,
                },
            ]
        )

    passed = bool(checks) and all(check["passed"] for check in checks)
    if passed and args.stage == "candidate":
        status = "candidate_gate_passed"
    else:
        status = "promoted" if passed else "rejected"
    return {
        "cycle": args.cycle,
        "timestamp": time.time(),
        "status": status,
        "checks": checks,
        "candidate_summary": args.candidate_summary,
        "feedback_meta": args.feedback_meta,
        "eval_summary": args.eval_summary,
        "model": args.model,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycle", default="residual-cycle")
    p.add_argument("--stage", choices=("candidate", "full"), default="full")
    p.add_argument("--candidate-summary", default="")
    p.add_argument("--feedback-meta", default="")
    p.add_argument("--eval-summary", default="")
    p.add_argument("--model", default="")
    p.add_argument("--previous-model", default="")
    p.add_argument(
        "--history",
        default="checkpoints/residual_evolution/history.jsonl",
    )
    p.add_argument(
        "--promotion-file",
        default="checkpoints/residual_evolution/promoted.json",
    )
    p.add_argument("--collect-cmd", default="")
    p.add_argument("--export-cmd", default="")
    p.add_argument("--train-cmd", default="")
    p.add_argument("--eval-cmd", default="")
    p.add_argument("--min-oracle-headroom", type=float, default=0.08)
    p.add_argument("--min-selector-accuracy", type=float, default=0.70)
    p.add_argument("--min-correctness-accuracy", type=float, default=0.70)
    p.add_argument("--max-action-regret", type=float, default=0.10)
    p.add_argument("--min-generation-valid", type=float, default=0.80)
    p.add_argument("--min-retention", type=float, default=0.98)
    p.add_argument("--min-paired-delta", type=float, default=0.05)
    p.add_argument("--min-ci-lower", type=float, default=0.0)
    p.add_argument("--max-error-rate", type=float, default=0.0)
    p.add_argument(
        "--restore-command-template",
        default="MERGED={model} bash scripts/serve_residual_gpu3.sh",
    )
    args = p.parse_args()

    promotion_path = Path(args.promotion_file)
    previous_model = args.previous_model
    if promotion_path.exists():
        previous_model = previous_model or str(
            json.loads(promotion_path.read_text(encoding="utf-8")).get("model")
            or ""
        )

    def restore_previous() -> None:
        if not previous_model or not args.restore_command_template:
            return
        command = args.restore_command_template.format(
            model=shlex.quote(previous_model)
        )
        _run("restore previous promoted model", command)

    try:
        _run("collect", args.collect_cmd)
        _run("export", args.export_cmd)
        _run("train", args.train_cmd)
        _run("eval", args.eval_cmd)
    except Exception:
        restore_previous()
        raise
    result = gate(args)
    history = Path(args.history)
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    if result["status"] == "promoted":
        promotion = promotion_path
        promotion.parent.mkdir(parents=True, exist_ok=True)
        promotion.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"PROMOTED: {args.model or '(model unspecified)'}")
    elif result["status"] == "rejected":
        print("REJECTED: previous promoted checkpoint remains active")
        restore_previous()
    else:
        print("CANDIDATE HEADROOM GATE PASSED")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
