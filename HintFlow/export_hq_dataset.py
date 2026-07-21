#!/usr/bin/env python3
"""Export a higher-quality HintFlow training set from existing trees.

Improvements over ``export_dpo_pairs.py`` (soft subtree-mean V + tau):

1. Terminal (hard) credit assignment. A node is scored by whether its subtree
   can actually *reach a correct terminal answer*, not by the soft mean of leaf
   EMs. Leaf EM (possibly a majority-vote fraction in v2 trees) is binarized to
   correct / wrong first.

2. Hard separation for preference pairs. A branching orch decision only becomes
   a DPO pair when one child reaches a correct terminal and the sibling reaches
   *zero* correct terminals (configurable). Same-terminal, near-margin review
   pairs (the noise that sank DPO v2) are dropped.

3. Sparse key decisions. Per tree we keep at most a few pairs, prioritising
   plan-level and shallow (more causal) decisions, and requiring a minimum leaf
   support on each side so 1-leaf estimates do not dominate.

4. Low-intervention supervision. Pairs whose chosen action is conservative
   (FINALIZE / NO_HINT / KEEP / NONE) while the rejected sibling intervened and
   failed are tagged, encoding the residual lesson "do not over-intervene".

5. Winning-trajectory SFT. For every tree that reaches a correct terminal we
   emit the orchestrator outputs along the most efficient correct path as SFT
   targets.

Outputs (under --out-dir):
  dpo_pairs.jsonl   preference pairs (schema compatible with train_dpo.py)
  sft.jsonl         winning-trajectory + low-intervention SFT targets
  summary.json      yield / quality statistics
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

from export_dpo_pairs import _prompt_for_child  # noqa: E402

CONSERVATIVE_ACTIONS = {"FINALIZE", "NO_HINT", "NONE", "KEEP"}
INTERVENTION_ACTIONS = {"INJECT", "REPLAN", "RETRY", "REVISE"}
_INVALID_ORCH = {"", "fallback"}


def _leaf_correct(em: Any, threshold: float) -> int | None:
    if em is None:
        return None
    try:
        return 1 if float(em) >= threshold else 0
    except (TypeError, ValueError):
        return None


def _valid_orch(node: dict) -> bool:
    raw = (node.get("orch_raw") or "").strip()
    if raw in _INVALID_ORCH:
        return False
    return node.get("kind") in {"plan", "review"}


def _action_of(node: dict) -> str | None:
    return (node.get("orch_parsed") or {}).get("action")


class TreeIndex:
    """Precomputes children, depth, and terminal-reach stats per node."""

    def __init__(self, tree: dict, *, leaf_threshold: float):
        self.tree = tree
        self.leaf_threshold = leaf_threshold
        nodes = tree.get("nodes") or []
        self.by_id: dict[str, dict] = {n["nid"]: n for n in nodes if n.get("nid")}
        self.children: dict[str, list[str]] = defaultdict(list)
        for n in nodes:
            pid = n.get("parent")
            if pid:
                self.children[pid].append(n["nid"])
        self.root = next((n["nid"] for n in nodes if n.get("kind") == "root"), None)
        self._depth: dict[str, int] = {}
        self._stats: dict[str, tuple[int, int]] = {}  # nid -> (n_correct, n_leaf)
        if self.root:
            self._compute_depth(self.root, 0)

    def _compute_depth(self, nid: str, d: int) -> None:
        self._depth[nid] = d
        for c in self.children.get(nid, []):
            self._compute_depth(c, d + 1)

    def depth(self, nid: str) -> int:
        return self._depth.get(nid, 10**6)

    def leaf_stats(self, nid: str) -> tuple[int, int]:
        """Return (n_correct, n_leaf) over the subtree rooted at nid."""
        if nid in self._stats:
            return self._stats[nid]
        node = self.by_id[nid]
        if node.get("is_leaf") or node.get("kind") == "leaf":
            lc = _leaf_correct(node.get("leaf_em"), self.leaf_threshold)
            res = (0, 0) if lc is None else (lc, 1)
            self._stats[nid] = res
            return res
        nc = nl = 0
        for c in self.children.get(nid, []):
            c_nc, c_nl = self.leaf_stats(c)
            nc += c_nc
            nl += c_nl
        self._stats[nid] = (nc, nl)
        return (nc, nl)

    def reach_rate(self, nid: str) -> float:
        nc, nl = self.leaf_stats(nid)
        return nc / nl if nl else 0.0


def _pair_from_parent(
    idx: TreeIndex,
    parent_nid: str,
    child_ids: list[str],
    *,
    problem: str,
    min_leaves: int,
    min_correct: int,
    mode: str,
    min_gap: float,
) -> dict[str, Any] | None:
    by_id = idx.by_id
    orch_kids = [by_id[c] for c in child_ids if c in by_id and _valid_orch(by_id[c])]
    if len(orch_kids) < 2:
        return None

    scored: list[tuple[dict, int, int]] = []
    for kid in orch_kids:
        nc, nl = idx.leaf_stats(kid["nid"])
        if nl < min_leaves:
            continue
        scored.append((kid, nc, nl))
    if len(scored) < 2:
        return None

    # best: highest reach_rate, tiebreak larger leaf support.
    scored.sort(key=lambda x: (x[1] / x[2], x[2]), reverse=True)
    chosen_n, c_nc, c_nl = scored[0]
    rejected_n, r_nc, r_nl = scored[-1]
    if chosen_n["nid"] == rejected_n["nid"]:
        return None

    c_rate = c_nc / c_nl
    r_rate = r_nc / r_nl
    if mode == "hard":
        if not (c_nc >= min_correct and r_nc == 0):
            return None
    else:  # "gap"
        if (c_rate - r_rate) < min_gap:
            return None

    if (chosen_n.get("orch_raw") or "").strip() == (rejected_n.get("orch_raw") or "").strip():
        return None

    try:
        task_c, system, user = _prompt_for_child(problem, chosen_n, by_id)
        task_r, system_r, user_r = _prompt_for_child(problem, rejected_n, by_id)
    except ValueError:
        return None
    # Same decision site only: siblings must share one orch prompt.
    if task_c != task_r or user != user_r or system != system_r:
        return None

    chosen_action = _action_of(chosen_n)
    rejected_action = _action_of(rejected_n)
    low_intervention = (
        chosen_action in CONSERVATIVE_ACTIONS
        and rejected_action in INTERVENTION_ACTIONS
    )

    depth = idx.depth(chosen_n["nid"])
    separation = c_rate - r_rate
    # Priority for sparse per-tree selection: plan first, shallow first,
    # stronger separation, larger support.
    priority = (
        1 if task_c == "plan" else 0,
        -depth,
        round(separation, 4),
        min(c_nl, r_nl),
    )

    return {
        "tree_id": idx.tree.get("id"),
        "gold": str(idx.tree.get("gold") or ""),
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
        "chosen_action": chosen_action,
        "rejected_action": rejected_action,
        "low_intervention": low_intervention,
        "depth": depth,
        "n_correct_chosen": c_nc,
        "n_leaves_chosen": c_nl,
        "n_correct_rejected": r_nc,
        "n_leaves_rejected": r_nl,
        "reach_rate_chosen": c_rate,
        "reach_rate_rejected": r_rate,
        "separation": separation,
        "credit": "terminal",
        "_priority": priority,
    }


def export_pairs_from_tree(
    idx: TreeIndex,
    *,
    min_leaves: int,
    min_correct: int,
    mode: str,
    min_gap: float,
    max_pairs_per_tree: int,
    max_review_per_tree: int,
) -> list[dict[str, Any]]:
    problem = idx.tree.get("problem") or ""
    candidates: list[dict[str, Any]] = []
    for parent_nid, child_ids in idx.children.items():
        rec = _pair_from_parent(
            idx,
            parent_nid,
            child_ids,
            problem=problem,
            min_leaves=min_leaves,
            min_correct=min_correct,
            mode=mode,
            min_gap=min_gap,
        )
        if rec is not None:
            candidates.append(rec)

    # Dedup identical (prompt, chosen, rejected).
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for rec in candidates:
        key = (rec["prompt"], rec["chosen"], rec["rejected"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    deduped.sort(key=lambda r: r["_priority"], reverse=True)

    kept: list[dict[str, Any]] = []
    n_review = 0
    for rec in deduped:
        if rec["task"] == "review":
            if n_review >= max_review_per_tree:
                continue
            n_review += 1
        kept.append(rec)
        if len(kept) >= max_pairs_per_tree:
            break

    for rec in kept:
        rec.pop("_priority", None)
    return kept


def _best_correct_leaf(idx: TreeIndex) -> str | None:
    """Correct leaf with fewest solver steps (most efficient winning path)."""
    best_nid = None
    best_steps = 10**9
    for nid, node in idx.by_id.items():
        if node.get("kind") != "leaf" and not node.get("is_leaf"):
            continue
        if _leaf_correct(node.get("leaf_em"), idx.leaf_threshold) != 1:
            continue
        steps = node.get("n_solver_steps")
        steps = steps if isinstance(steps, int) else 10**8
        if steps < best_steps:
            best_steps = steps
            best_nid = nid
    return best_nid


def export_sft_from_tree(idx: TreeIndex) -> list[dict[str, Any]]:
    leaf_nid = _best_correct_leaf(idx)
    if not leaf_nid:
        return []
    problem = idx.tree.get("problem") or ""
    # Walk root -> leaf, collect orch nodes on the path.
    chain: list[dict] = []
    nid: str | None = leaf_nid
    seen: set[str] = set()
    while nid and nid in idx.by_id and nid not in seen:
        seen.add(nid)
        chain.append(idx.by_id[nid])
        nid = idx.by_id[nid].get("parent")
    chain.reverse()

    samples: list[dict[str, Any]] = []
    for node in chain:
        if not _valid_orch(node):
            continue
        try:
            task, system, user = _prompt_for_child(problem, node, idx.by_id)
        except ValueError:
            continue
        samples.append(
            {
                "tree_id": idx.tree.get("id"),
                "gold": str(idx.tree.get("gold") or ""),
                "nid": node["nid"],
                "task": task,
                "system": system,
                "prompt": user,
                "target": node.get("orch_raw") or "",
                "action": _action_of(node),
                "depth": idx.depth(node["nid"]),
                "source": "winning_path",
            }
        )
    return samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trees",
        default=",".join(
            [
                str(_ROOT / "checkpoints" / "hintflow_trees_v2" / "trees.jsonl"),
                str(_ROOT / "checkpoints" / "hintflow_trees_2k" / "trees.jsonl"),
            ]
        ),
        help="comma-separated tree jsonl files",
    )
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "checkpoints" / "hintflow_hq_dataset"),
    )
    p.add_argument("--leaf-threshold", type=float, default=0.5)
    p.add_argument(
        "--mode",
        choices=("hard", "gap"),
        default="hard",
        help="hard: chosen reaches correct & rejected reaches zero; gap: reach-rate gap",
    )
    p.add_argument("--min-gap", type=float, default=0.5, help="only used in gap mode")
    p.add_argument("--min-leaves", type=int, default=2)
    p.add_argument("--min-correct", type=int, default=1)
    p.add_argument("--max-pairs-per-tree", type=int, default=3)
    p.add_argument("--max-review-per-tree", type=int, default=2)
    p.add_argument("--no-sft", action="store_true", help="skip winning-trajectory SFT export")
    args = p.parse_args()

    tree_files = [Path(x.strip()) for x in args.trees.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "dpo_pairs.jsonl"
    sft_path = out_dir / "sft.jsonl"

    n_trees = n_valid = 0
    by_task: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    n_pairs = n_low_int = n_sft = 0
    seps: list[float] = []
    sft_actions: dict[str, int] = defaultdict(int)
    pair_chosen_actions: dict[str, int] = defaultdict(int)

    with pairs_path.open("w", encoding="utf-8") as fpair, (
        sft_path.open("w", encoding="utf-8") if not args.no_sft else _Null()
    ) as fsft:
        for tf in tree_files:
            if not tf.exists():
                print(f"WARN missing tree file: {tf}", flush=True)
                continue
            with tf.open(encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    tree = json.loads(line)
                    n_trees += 1
                    if tree.get("error"):
                        continue
                    n_valid += 1
                    idx = TreeIndex(tree, leaf_threshold=args.leaf_threshold)

                    pairs = export_pairs_from_tree(
                        idx,
                        min_leaves=args.min_leaves,
                        min_correct=args.min_correct,
                        mode=args.mode,
                        min_gap=args.min_gap,
                        max_pairs_per_tree=args.max_pairs_per_tree,
                        max_review_per_tree=args.max_review_per_tree,
                    )
                    for rec in pairs:
                        fpair.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_pairs += 1
                        by_task[rec["task"]] += 1
                        seps.append(rec["separation"])
                        if rec["low_intervention"]:
                            n_low_int += 1
                        pair_chosen_actions[str(rec.get("chosen_action"))] += 1

                    if not args.no_sft:
                        sft = export_sft_from_tree(idx)
                        for s in sft:
                            fsft.write(json.dumps(s, ensure_ascii=False) + "\n")
                            n_sft += 1
                            by_source[s["source"]] += 1
                            sft_actions[str(s.get("action"))] += 1

    summary = {
        "tree_files": [str(x) for x in tree_files],
        "trees_seen": n_trees,
        "trees_valid": n_valid,
        "config": {
            "leaf_threshold": args.leaf_threshold,
            "mode": args.mode,
            "min_gap": args.min_gap,
            "min_leaves": args.min_leaves,
            "min_correct": args.min_correct,
            "max_pairs_per_tree": args.max_pairs_per_tree,
            "max_review_per_tree": args.max_review_per_tree,
        },
        "dpo_pairs": {
            "n": n_pairs,
            "by_task": dict(by_task),
            "low_intervention": n_low_int,
            "chosen_action_dist": dict(pair_chosen_actions),
            "separation_mean": (sum(seps) / len(seps)) if seps else 0.0,
            "separation_min": min(seps) if seps else 0.0,
            "out": str(pairs_path),
        },
        "sft": {
            "n": n_sft,
            "by_source": dict(by_source),
            "action_dist": dict(sft_actions),
            "out": str(sft_path) if not args.no_sft else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


class _Null:
    """No-op context manager / file when SFT export is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, *_a, **_k):
        return None


if __name__ == "__main__":
    main()
