#!/usr/bin/env python3
"""Export offline DPO pairs from HintFlow trees.

Each branching parent with >=2 orch children (plan/review) yields at most one pair:
  prompt = problem + prefix (rebuild to match HintFlowAgent orch input)
  chosen/rejected = child orch_raw (prefer higher subtree leaf-EM mean)
  drop if |V_chosen - V_rejected| < tau
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from HintFlowAgent import PLAN_SYSTEM, REVIEW_SYSTEM  # noqa: E402


def _subtree_leaf_ems(nid: str, children: dict[str, list[str]], by_id: dict[str, dict]) -> list[float]:
    node = by_id[nid]
    if node.get("is_leaf") or node.get("kind") == "leaf":
        em = node.get("leaf_em")
        return [float(em)] if em is not None else []
    out: list[float] = []
    for c in children.get(nid, []):
        out.extend(_subtree_leaf_ems(c, children, by_id))
    return out


def _v_mean(ems: list[float]) -> float | None:
    if not ems:
        return None
    return sum(ems) / len(ems)


def _ancestors(nid: str | None, by_id: dict[str, dict]) -> list[dict]:
    chain: list[dict] = []
    seen: set[str] = set()
    while nid and nid in by_id and nid not in seen:
        seen.add(nid)
        node = by_id[nid]
        chain.append(node)
        nid = node.get("parent")
    return chain  # [node, parent, ..., root]


def _rebuild_plan_prompt(problem: str, context: str) -> tuple[str, str]:
    user = f"Problem:\n{problem}"
    if (context or "").strip():
        user += f"\n\nContext from prior execution (replan):\n{context.strip()}"
    return PLAN_SYSTEM, user


def _rebuild_review_prompt(
    problem: str,
    *,
    instruction: str,
    observation: str,
    running_summary: str,
    hist: str,
    inject_prior: bool,
    is_last_planned: bool,
) -> tuple[str, str]:
    user = (
        f"Problem:\n{problem}\n\n"
        f"Running summary:\n{running_summary or '(empty)'}\n\n"
        f"Recent reviews:\n{hist or '(none)'}\n\n"
        f"Current instruction:\n{instruction}\n\n"
        f"Solver response:\n{observation}\n\n"
        f"inject_prior (soft): {str(inject_prior).lower()}\n"
        f"is_last_planned_node: {str(is_last_planned).lower()}\n\n"
        "Return the JSON review now."
    )
    return REVIEW_SYSTEM, user


def _history_from_path(child: dict, by_id: dict[str, dict]) -> str:
    """Approximate Recent reviews from ancestor review nodes on the path."""
    # path excluding the child itself: parent ... root
    chain = _ancestors(child.get("parent"), by_id)
    reviews = [n for n in reversed(chain) if n.get("kind") == "review"]
    # last up to 4 reviews before current
    reviews = reviews[-4:]
    if not reviews:
        return "(none)"
    lines: list[str] = []
    for n in reviews:
        parsed = n.get("orch_parsed") or {}
        instr = (n.get("instruction") or "")[:160]
        step_i = n.get("step_index")
        label = f"step{(step_i + 1) if isinstance(step_i, int) else '?'}"
        lines.append(f"- {label}: {instr}")
        if parsed.get("summary"):
            lines.append(f"  summary: {str(parsed['summary'])[:200]}")
        if parsed.get("issue"):
            lines.append(f"  issue: {str(parsed['issue'])[:160]}")
    return "\n".join(lines) if lines else "(none)"


def _infer_is_last_planned(child: dict, by_id: dict[str, dict]) -> bool:
    step_i = child.get("step_index")
    if not isinstance(step_i, int):
        return False
    for n in _ancestors(child.get("parent"), by_id):
        if n.get("kind") == "plan":
            nodes = (n.get("orch_parsed") or {}).get("nodes") or []
            if nodes:
                return step_i >= len(nodes) - 1
            break
    return False


def _prompt_for_child(problem: str, child: dict, by_id: dict[str, dict]) -> tuple[str, str, str]:
    """Return (task_kind, system, user) matching HintFlowAgent orch calls."""
    kind = child.get("kind")
    # v2 trees store the exact clean prompt (turn budget included, contrastive
    # block excluded) — always prefer it over the lossy rebuild below.
    train_prompt = child.get("train_prompt")
    if train_prompt and kind in {"plan", "review"}:
        system = PLAN_SYSTEM if kind == "plan" else REVIEW_SYSTEM
        return kind, system, train_prompt
    if kind == "plan":
        system, user = _rebuild_plan_prompt(problem, child.get("context") or "")
        return "plan", system, user
    if kind == "review":
        system, user = _rebuild_review_prompt(
            problem,
            instruction=child.get("instruction") or "",
            observation=child.get("observation") or "",
            running_summary=child.get("running_summary_before") or "",
            hist=_history_from_path(child, by_id),
            inject_prior=False,  # not stored on tree nodes
            is_last_planned=_infer_is_last_planned(child, by_id),
        )
        return "review", system, user
    raise ValueError(f"unsupported kind {kind}")


def export_tree(
    tree: dict[str, Any],
    *,
    tau: float,
    min_leaves: int,
) -> list[dict[str, Any]]:
    if tree.get("error"):
        return []
    problem = tree.get("problem") or ""
    tree_id = tree.get("id")
    nodes = tree.get("nodes") or []
    by_id = {n["nid"]: n for n in nodes if n.get("nid")}
    children: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        pid = n.get("parent")
        if pid:
            children[pid].append(n["nid"])

    # cache V
    v_cache: dict[str, float | None] = {}
    nleaf_cache: dict[str, int] = {}

    def v_of(nid: str) -> float | None:
        if nid in v_cache:
            return v_cache[nid]
        ems = _subtree_leaf_ems(nid, children, by_id)
        nleaf_cache[nid] = len(ems)
        v_cache[nid] = _v_mean(ems)
        return v_cache[nid]

    pairs: list[dict[str, Any]] = []
    for parent_nid, child_ids in children.items():
        orch_kids = [
            by_id[c]
            for c in child_ids
            if c in by_id
            and by_id[c].get("kind") in {"plan", "review"}
            and (by_id[c].get("orch_raw") or "").strip()
        ]
        if len(orch_kids) < 2:
            continue

        # k=2 typical; if more, use best vs worst by V
        scored: list[tuple[dict, float, int]] = []
        for kid in orch_kids:
            vv = v_of(kid["nid"])
            if vv is None:
                continue
            nl = nleaf_cache[kid["nid"]]
            if nl < min_leaves:
                continue
            scored.append((kid, vv, nl))
        if len(scored) < 2:
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_n, v_c, nl_c = scored[0]
        rejected_n, v_r, nl_r = scored[-1]
        if chosen_n["nid"] == rejected_n["nid"]:
            continue
        if abs(v_c - v_r) < tau:
            continue
        # identical outputs: useless
        if (chosen_n.get("orch_raw") or "").strip() == (rejected_n.get("orch_raw") or "").strip():
            continue

        # Prompt from chosen child's state; siblings should share the same decision context.
        # Prefer chosen's rebuild; assert roughly same kind as rejected for logging.
        try:
            task_c, system, user = _prompt_for_child(problem, chosen_n, by_id)
            task_r, system_r, user_r = _prompt_for_child(problem, rejected_n, by_id)
        except ValueError:
            continue

        # If kinds differ, still one prompt — use the higher-V child's template, but keep both
        # responses. User wants learning "what action type in what state"; store both tasks.
        # For DPO the prompt must be shared: if templates differ, skip (can't share one prompt).
        if task_c != task_r or user != user_r or system != system_r:
            # Different decision templates → not the same orch call site; skip.
            continue

        pairs.append(
            {
                "tree_id": tree_id,
                "parent_nid": parent_nid,
                "task": task_c,
                "system": system,
                "prompt": user,
                "chosen": chosen_n.get("orch_raw") or "",
                "rejected": rejected_n.get("orch_raw") or "",
                "chosen_parsed": chosen_n.get("orch_parsed") or {},
                "rejected_parsed": rejected_n.get("orch_parsed") or {},
                "chosen_nid": chosen_n["nid"],
                "rejected_nid": rejected_n["nid"],
                "v_chosen": v_c,
                "v_rejected": v_r,
                "n_leaves_chosen": nl_c,
                "n_leaves_rejected": nl_r,
                "delta_v": v_c - v_r,
                "truncated_tree": bool(tree.get("truncated")),
            }
        )
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trees",
        default=str(_ROOT / "checkpoints" / "hintflow_trees_2k" / "trees.jsonl"),
    )
    p.add_argument(
        "--out",
        default=str(_ROOT / "checkpoints" / "hintflow_trees_2k" / "dpo_pairs.jsonl"),
    )
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument(
        "--min-leaves",
        type=int,
        default=1,
        help="min leaf count on each side (default 1; raise if truncated V too noisy)",
    )
    p.add_argument("--limit-trees", type=int, default=0)
    args = p.parse_args()

    trees_path = Path(args.trees)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_trees = n_pairs = n_skip_trees = 0
    by_task: dict[str, int] = defaultdict(int)
    deltas: list[float] = []

    with trees_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            tree = json.loads(line)
            n_trees += 1
            if args.limit_trees and n_trees > args.limit_trees:
                break
            pairs = export_tree(tree, tau=args.tau, min_leaves=args.min_leaves)
            if not pairs:
                n_skip_trees += 1
            for rec in pairs:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pairs += 1
                by_task[rec["task"]] += 1
                deltas.append(rec["delta_v"])

    summary = {
        "trees": n_trees,
        "trees_with_zero_pairs": n_skip_trees,
        "n_pairs": n_pairs,
        "by_task": dict(by_task),
        "tau": args.tau,
        "min_leaves": args.min_leaves,
        "delta_v_mean": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "out": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
