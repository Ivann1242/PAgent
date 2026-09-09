"""Question-paired summaries. No model imports; plots use measured data only."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics

from .io import dataset, file_hash, rows, write_json


def paired_ci(values, repeats=2000, seed=712):
    if not values:
        raise ValueError("Cannot estimate an empty comparison")
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [draws[int(.025 * repeats)], draws[min(int(.975 * repeats), repeats - 1)]]


def usage(paths):
    totals = {"requests": 0, "requested_output_cap": 0, "prompt_tokens": 0,
              "completion_tokens": 0, "missing_usage": 0, "elapsed_sec": 0.0,
              "errors": 0, "unresolved_requests": 0}
    requests, responses = set(), set()
    roles = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                 "missing_usage": 0, "elapsed_sec": 0.0})
    for path in sorted(set(paths)):
        if not path.exists():
            continue
        for record in rows(path):
            cid = record["call_id"]
            role = roles[record.get("role", "unknown")]
            if record["event"] == "request":
                requests.add(cid)
                totals["requests"] += 1
                role["requests"] += 1
                totals["requested_output_cap"] += record["request"].get("max_tokens", 0)
            else:
                responses.add(cid)
                totals["elapsed_sec"] += record.get("elapsed_sec", 0)
                role["elapsed_sec"] += record.get("elapsed_sec", 0)
                if record["event"] != "response":
                    totals["errors"] += 1
                if record.get("usage") is None:
                    totals["missing_usage"] += 1
                    role["missing_usage"] += 1
                else:
                    for key in ("prompt_tokens", "completion_tokens"):
                        totals[key] += record["usage"].get(key) or 0
                        role[key] += record["usage"].get(key) or 0
    totals["unresolved_requests"] = len(requests - responses)
    # Token totals with missing usage are lower bounds, never zero-cost errors.
    totals["tokens_complete"] = not (totals["missing_usage"] or totals["unresolved_requests"])
    totals["by_role"] = dict(roles)
    return totals


def analyze(output, plots=False):
    output = Path(output).resolve()
    cfg = json.loads((output / "config.json").read_text())
    expected_ids = {str(row["id"]) for row in dataset(cfg["eval_data"])}
    records = []
    sources = {}
    for path in sorted((output / "eval" / "results").glob("*/*/*/*.json")):
        row = json.loads(path.read_text())
        row["record_path"] = str(path)
        records.append(row)
        sources[str(path)] = file_hash(path)
    if not records:
        raise ValueError("No evaluation results yet")
    destination = output / "analysis"
    destination.mkdir(exist_ok=True)
    fields = ["id", "eval_seed", "train_seed", "method", "selector", "em", "baseline_em",
              "challenger_em", "union", "random_expected_em", "recovered", "harmed", "error"]
    with (destination / "paired_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    groups = defaultdict(list)
    for row in records:
        groups[(row["method"], row["selector"])].append(row)
    summary, question_means = {}, {}
    for (method, selector), items in groups.items():
        name = f"{method}/{selector}"
        expected_seeds = cfg["seeds"] if method in {"vp", "off", "sft_swap"} else [cfg["seeds"][0]]
        expected = {(qid, es, ts) for qid in expected_ids for es in cfg["eval_seeds"] for ts in expected_seeds}
        observed = {(str(r["id"]), r["eval_seed"], r["train_seed"]) for r in items}
        if len(observed) != len(items):
            raise ValueError(f"Duplicate trial keys for {name}")
        complete = observed == expected
        by_question = defaultdict(list)
        for row in items:
            by_question[str(row["id"])].append(row["em"])
            if row["em"] - row["baseline_em"] != row["recovered"] - row["harmed"]:
                raise ValueError("Recovery/harm identity failed")
        means = {qid: statistics.mean(values) for qid, values in by_question.items()}
        question_means[name] = means
        per_seed = {str(seed): statistics.mean(r["em"] for r in items if r["train_seed"] == seed)
                    for seed in sorted({r["train_seed"] for r in items})}
        summary[name] = {"complete": complete, "observed": len(observed), "expected": len(expected),
                         "n_questions": len(means), "em": statistics.mean(means.values()),
                         "question_bootstrap_ci": paired_ci(list(means.values())) if complete else None,
                         "per_train_seed_em": per_seed,
                         "metrics": {k: statistics.mean(r[k] for r in items) for k in
                                     ("baseline_em", "challenger_em", "union", "random_expected_em", "recovered", "harmed")},
                         "candidate_errors": sum(bool(r["error"]) for r in items),
                         "selector_failures": sum("selector failure" in (r.get("selection") or {}).get("reason", "") for r in items)}
    comparisons = {}
    for left, right in (("vp/sft", "off/sft"), ("vp/sft", "sft/sft"), ("vp/sft", "resample/sft"),
                        ("vp/sft", "fixed_cot/sft"), ("vp/current", "vp/sft"), ("sft_swap/current", "sft/sft")):
        if left not in summary or right not in summary or not (summary[left]["complete"] and summary[right]["complete"]):
            continue
        differences = [question_means[left][qid] - question_means[right][qid] for qid in sorted(expected_ids)]
        comparisons[f"{left} - {right}"] = {"difference": statistics.mean(differences),
                                            "paired_question_bootstrap_ci": paired_ci(differences),
                                            "n_questions": len(differences)}
        l_seeds, r_seeds = summary[left]["per_train_seed_em"], summary[right]["per_train_seed_em"]
        comparisons[f"{left} - {right}"]["per_train_seed_difference"] = {
            seed: value - (r_seeds[seed] if seed in r_seeds else next(iter(r_seeds.values())))
            for seed, value in l_seeds.items() if seed in r_seeds or len(r_seeds) == 1}
    write_json(destination / "statistics.json", {"methods": summary, "comparisons": comparisons,
               "ci_scope": "Question sampling conditional on observed training/evaluation seeds; not full pipeline training uncertainty",
               "multiplicity": "No significance decisions; VP-on vs off is primary, other comparisons descriptive",
               "metric_scale": "fraction", "sources": sources})
    # Both physical workload (unique journal files) and deployment-equivalent
    # per-method cost (bare charged again to each trial) remain auditable.
    cost = {"physical_workload": usage(list(output.rglob("*.calls.jsonl")) + list((output / "train").glob("*/*/attempts/*/calls.jsonl"))),
            "by_method": {}}
    for (method, selector), items in groups.items():
        totals = []
        for row in items:
            paths = [Path(row["baseline_source"]).with_suffix(".calls.jsonl"),
                     Path(row["candidate_source"]).with_suffix(".calls.jsonl"),
                     Path(row["record_path"]).with_suffix(".calls.jsonl")]
            totals.append(usage(paths))
        cost["by_method"][f"{method}/{selector}"] = {
            "mean_per_trial": {k: statistics.mean(t[k] for t in totals) for k in
                               ("requests", "requested_output_cap", "prompt_tokens", "completion_tokens", "elapsed_sec")},
            "tokens_complete": all(t["tokens_complete"] for t in totals),
            "latency_scope": "Sum of actual service times for cached components; not a fresh deployment wall-clock measurement"}
    write_json(destination / "costs.json", cost)
    training = {}
    for path in sorted((output / "train").glob("*/*/steps.jsonl")):
        # Keep the latest replay of each step, only through an immutable committed snapshot.
        from .runner import latest_checkpoint
        checkpoint = latest_checkpoint(path.parent)
        committed = json.loads((checkpoint / "grpo_train_state.json").read_text())["step"] if checkpoint else 0
        dedup = {r["step"]: r for r in rows(path) if r["step"] <= committed}
        training[str(path.relative_to(output))] = [dedup[k] for k in sorted(dedup)]
    write_json(destination / "training_curves.json", training)
    hint_lengths = defaultdict(list)
    for path in sorted((output / "eval" / "candidates").glob("*/*/*.json")):
        if path.name.endswith(".partial.json"):
            continue
        item = json.loads(path.read_text())
        if "hint_tokens" in item:
            hint_lengths[path.parent.parent.name].append(item["hint_tokens"])
    write_json(destination / "hint_lengths.json", dict(hint_lengths))
    budget_summary = analyze_budgets(output, cfg, expected_ids)
    write_json(destination / "budget_statistics.json", budget_summary)
    if plots:
        plot(destination, summary, training)
        plot_budgets(destination, budget_summary)
        plot_lengths(destination, hint_lengths)
    print(f"Analysis written to {destination}; incomplete methods have no CI")


def plot_lengths(destination, groups):
    if not groups:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for name, values in sorted(groups.items()):
        values = sorted(values)
        ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post", label=name)
    ax.set_xlabel("Hint tokens (policy tokenizer)")
    ax.set_ylabel("Empirical CDF across generated hints")
    ax.legend()
    fig.savefig(destination / "hint_lengths.pdf")
    fig.savefig(destination / "hint_lengths.png", dpi=180)
    plt.close(fig)


def analyze_budgets(output, cfg, expected_ids):
    summary = {}
    seed = cfg["seeds"][0]
    expected = {(qid, es) for qid in expected_ids for es in cfg["eval_seeds"]}
    for budget in cfg["budgets"]:
        root = output / f"budget-{budget}"
        points = {}
        paired = {}
        for method, paths in (
            ("hf1", sorted((root / "results" / "vp" / str(seed) / "current").glob("*.json"))),
            ("hfm", sorted((root / "hfm").glob("*.json"))),
        ):
            paths = [p for p in paths if not p.name.endswith(".partial.json")]
            if not paths:
                continue
            samples = [json.loads(p.read_text()) for p in paths]
            observed = {(str(r["id"]), r["eval_seed"]) for r in samples}
            if len(observed) != len(samples):
                raise ValueError("Duplicate budget trial keys")
            complete = observed == expected
            costs, by_q = [], defaultdict(list)
            for path, row in zip(paths, samples):
                by_q[str(row["id"])].append(row["em"])
                journals = [path.with_suffix(".calls.jsonl")]
                if method == "hf1":
                    journals += [Path(row[k]).with_suffix(".calls.jsonl") for k in ("baseline_source", "candidate_source")]
                costs.append(usage(journals))
            means = {qid: statistics.mean(v) for qid, v in by_q.items()}
            paired[method] = means
            points[method] = {
                "complete": complete, "trials": len(samples), "expected": len(expected),
                "em": statistics.mean(means.values()),
                "question_bootstrap_ci": paired_ci(list(means.values())) if complete else None,
                "tokens_complete": all(c["tokens_complete"] for c in costs),
                "mean_solver_output_tokens": statistics.mean(c["by_role"].get("solver", {}).get("completion_tokens", 0) for c in costs),
                "mean_total_input_tokens": statistics.mean(c["prompt_tokens"] for c in costs),
                "mean_total_output_tokens": statistics.mean(c["completion_tokens"] for c in costs),
                "mean_service_elapsed_sec": statistics.mean(c["elapsed_sec"] for c in costs),
                "errors": sum(bool(r.get("error")) for r in samples),
                "draft_missing": sum(bool(r.get("draft_missing")) for r in samples) if method == "hfm" else None,
                "plan_lengths": [len(r["plan"]) for r in samples if "plan" in r] if method == "hfm" else None,
            }
        if all(k in points and points[k]["complete"] for k in ("hf1", "hfm")):
            difference = [paired["hfm"][qid] - paired["hf1"][qid] for qid in sorted(expected_ids)]
            points["paired_hfm_minus_hf1"] = {"difference": statistics.mean(difference), "ci": paired_ci(difference)}
        if points:
            summary[str(budget)] = points
    return summary


def plot_budgets(destination, summary):
    if not summary:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for method, color in (("hf1", "#1b76d2"), ("hfm", "#483e8c")):
        points = [(int(budget), record[method]) for budget, record in summary.items()
                  if method in record and record[method]["complete"] and record[method]["tokens_complete"]]
        points.sort()
        if not points:
            continue
        for ax, key in zip(axes, ("mean_solver_output_tokens", "mean_service_elapsed_sec")):
            ax.plot([p[key] for _, p in points], [p["em"] * 100 for _, p in points], "o-", color=color, label=method)
            for budget, point in points:
                ax.annotate(str(budget), (point[key], point["em"] * 100), fontsize=7)
    axes[0].set_xlabel("Actual solver output tokens per problem")
    axes[1].set_xlabel("Sum of measured service seconds (cached HF1 components)")
    for ax in axes:
        ax.set_ylabel("Final EM (%)")
        if ax.lines:
            ax.legend()
    fig.savefig(destination / "budget.pdf")
    fig.savefig(destination / "budget.png", dpi=180)
    plt.close(fig)


def plot(destination, summary, training):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    complete = [(k, v) for k, v in summary.items() if v["complete"]]
    if complete:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for i, (name, value) in enumerate(complete):
            mean = value["em"] * 100
            lo, hi = [v * 100 for v in value["question_bootstrap_ci"]]
            axes[0].plot([i, i], [lo, hi], color="#1b76d2")
            axes[0].plot(i, mean, "o", color="#1b76d2")
            axes[1].bar(i - .15, value["metrics"]["recovered"] * 100, width=.3, color="#483e8c")
            axes[1].bar(i + .15, -value["metrics"]["harmed"] * 100, width=.3, color="#dc8969")
        for ax in axes:
            ax.set_xticks(range(len(complete)), [x[0] for x in complete], rotation=60, ha="right")
        axes[0].set_ylabel("Final EM (%) / question bootstrap 95% CI")
        axes[1].set_ylabel("Recovery (+) / harm (-), percentage points")
        fig.savefig(destination / "main_results.pdf")
        fig.savefig(destination / "main_results.png", dpi=180)
        plt.close(fig)
    if training:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for name, records in training.items():
            cumulative, x, y = 0, [], []
            for row in records:
                cumulative += row["feedback_jobs"]
                total = sum(row["group_types"].values())
                x.append(cumulative)
                y.append(row["group_types"]["all_wrong"] / max(total, 1))
            ax.plot(x, y, label=name.replace("/steps.jsonl", ""))
        ax.set_xlabel("Feedback jobs in committed updates (excludes discarded attempts)")
        ax.set_ylabel("All-wrong group fraction (includes empty in denominator)")
        ax.legend(fontsize=7)
        fig.savefig(destination / "training_groups.pdf")
        fig.savefig(destination / "training_groups.png", dpi=180)
        plt.close(fig)
