#!/usr/bin/env python3
"""Offline HintFlow tree collection (k-branch at every orch call).

Saves one JSONL record per problem with the full decision tree and leaf EMs.
Value / advantage are left for post-processing.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core import exact_match, extract_final_answer, has_parseable_answer, load_jsonl  # noqa: E402
from HintFlowAgent import (  # noqa: E402
    ORCH_MODEL,
    ORCH_URL,
    SOLVER_MODEL,
    HintFlowAgent,
    PlanNode,
    StepRecord,
    StepReview,
    _message_text,
    build_plan_user,
    build_review_user,
    build_solver_user_message,
    format_history,
    format_turn_info,
)

DEFAULT_SOLVER_URLS = [
    "http://127.0.0.1:8006/v1",
    "http://127.0.0.1:8007/v1",
    "http://127.0.0.1:8008/v1",
    "http://127.0.0.1:8009/v1",
]

# Cap in-flight OSS calls across problem workers + sibling forks.
# Sized to aggregate vLLM capacity (n_servers * max-num-seqs); reset in main().
_SOLVER_SEM = threading.Semaphore(32)
_PRINT_LOCK = threading.Lock()


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


class _SolverPool:
    """Least-loaded routing over all solver instances (stateless HTTP, so any
    call can go to any server). Replaces sticky per-problem routing; slower
    servers (e.g. sharing a GPU with other jobs) naturally receive fewer calls."""

    def __init__(self, urls: list[str], model: str):
        self._clients = [OpenAI(base_url=u, api_key="EMPTY") for u in urls]
        self._model = model
        self._inflight = [0] * len(self._clients)
        self._i = 0
        self._lock = threading.Lock()

    def _acquire(self) -> int:
        with self._lock:
            lo = min(self._inflight)
            # round-robin among the least-loaded to avoid always hitting idx 0
            n = len(self._clients)
            for off in range(n):
                idx = (self._i + off) % n
                if self._inflight[idx] == lo:
                    break
            self._i = idx + 1
            self._inflight[idx] += 1
            return idx

    def _release(self, idx: int) -> None:
        with self._lock:
            self._inflight[idx] -= 1

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        idx = self._acquire()
        try:
            resp = self._clients[idx].chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return _message_text(resp)
        finally:
            self._release(idx)


@dataclass
class PathState:
    plan_nodes: list[PlanNode]
    node_i: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    running_summary: str = ""
    pending_inject: str = ""
    retries_on_node: int = 0
    step_counter: int = 0


@dataclass
class TreeBuilder:
    agent: HintFlowAgent
    problem: str
    gold: str
    problem_id: Any = None
    k: int = 2
    max_steps: int = 7  # hard depth cap (solver turns per path); last turn forces answer
    max_solver_calls: int = 0  # 0 = unlimited (safety valve only)
    max_leaves: int = 0  # 0 = unlimited (safety valve only)
    obs_truncate: int = 6000
    contrastive: bool = True  # sibling j>0 sees sibling outputs (collection only)
    leaf_em_samples: int = 3  # terminal answer re-sampled to de-noise EM
    leaf_resample_temp: float = 0.7
    solver_pool: _SolverPool | None = None  # per-call RR; falls back to agent.solver

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_solver_calls: int = 0
    n_leaves: int = 0
    truncated: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _nid(self) -> str:
        return uuid.uuid4().hex[:10]

    def _trunc(self, text: str) -> str:
        if len(text) <= self.obs_truncate:
            return text
        return text[: self.obs_truncate] + "\n...[truncated]..."

    def _add(
        self,
        *,
        kind: str,
        parent: str | None,
        orch_raw: str = "",
        orch_parsed: dict | None = None,
        extra: dict | None = None,
    ) -> str:
        nid = self._nid()
        rec: dict[str, Any] = {
            "nid": nid,
            "parent": parent,
            "kind": kind,
            "orch_raw": orch_raw,
            "orch_parsed": orch_parsed or {},
            "children": [],
            "is_leaf": False,
            "leaf_em": None,
            "final_answer": None,
            "truncated": False,
        }
        if extra:
            rec.update(extra)
        with self._lock:
            self.nodes[nid] = rec
            if parent is not None and parent in self.nodes:
                self.nodes[parent]["children"].append(nid)
        return nid

    def _make_leaf(
        self,
        parent: str | None,
        state: PathState,
        *,
        reason: str,
        observation: str = "",
    ) -> str:
        obs = observation or (state.steps[-1].observation if state.steps else "")
        pred = extract_final_answer(obs) if obs else ""
        em0 = exact_match(pred, self.gold) if self.gold else 0

        # De-noise terminal EM: re-sample the final solver turn (T>0) and average.
        ems = [int(em0)]
        can_resample = (
            self.leaf_em_samples > 1
            and self.gold
            and not reason.startswith("cap_")
            and state.messages
            and state.messages[-1].get("role") == "assistant"
            and state.messages[-1].get("content") == obs
        )
        if can_resample:
            base_msgs = state.messages[:-1]
            for _ in range(self.leaf_em_samples - 1):
                try:
                    with _SOLVER_SEM:
                        if self.solver_pool is not None:
                            alt = self.solver_pool.chat(
                                base_msgs,
                                temperature=self.leaf_resample_temp,
                                max_tokens=self.agent.solver_max_tokens,
                            )
                        else:
                            resp = self.agent.solver.chat.completions.create(
                                model=self.agent.solver_model,
                                messages=base_msgs,
                                temperature=self.leaf_resample_temp,
                                max_tokens=self.agent.solver_max_tokens,
                            )
                            alt = (resp.choices[0].message.content or "").strip()
                except Exception:
                    continue
                with self._lock:
                    self.n_solver_calls += 1
                ems.append(exact_match(extract_final_answer(alt), self.gold))

        leaf_em = sum(ems) / len(ems)
        nid = self._add(
            kind="leaf",
            parent=parent,
            extra={
                "is_leaf": True,
                "leaf_em": leaf_em,
                "leaf_em_votes": ems,
                "final_answer": pred,
                "n_solver_steps": state.step_counter,
                "leaf_reason": reason,
                "observation": self._trunc(obs),
                "running_summary": state.running_summary,
                "truncated": reason.startswith("cap_"),
            },
        )
        with self._lock:
            self.n_leaves += 1
            if reason.startswith("cap_"):
                self.truncated = True
        return nid

    def _budget_ok(self) -> bool:
        with self._lock:
            if self.max_solver_calls > 0 and self.n_solver_calls >= self.max_solver_calls:
                self.truncated = True
                return False
            if self.max_leaves > 0 and self.n_leaves >= self.max_leaves:
                self.truncated = True
                return False
        return True

    def _reserve_solver(self) -> bool:
        """Atomically claim one solver slot before calling OSS."""
        with self._lock:
            if self.max_solver_calls > 0 and self.n_solver_calls >= self.max_solver_calls:
                self.truncated = True
                return False
            if self.max_leaves > 0 and self.n_leaves >= self.max_leaves:
                self.truncated = True
                return False
            self.n_solver_calls += 1
            ncall = self.n_solver_calls
            nleaf = self.n_leaves
        if ncall == 1 or ncall % 4 == 0:
            with _PRINT_LOCK:
                print(
                    f"[tree] id={self.problem_id} solver={ncall} leaves={nleaf}",
                    flush=True,
                )
        return True

    def _clone_state(self, state: PathState) -> PathState:
        return PathState(
            plan_nodes=[
                PlanNode(n.instruction, n.inject_after, n.is_final) for n in state.plan_nodes
            ],
            node_i=state.node_i,
            messages=copy.deepcopy(state.messages),
            steps=copy.deepcopy(state.steps),
            running_summary=state.running_summary,
            pending_inject=state.pending_inject,
            retries_on_node=state.retries_on_node,
            step_counter=state.step_counter,
        )

    def _solver_once(self, state: PathState) -> str | None:
        """Run one solver step. Returns None if budget exhausted."""
        if state.node_i >= len(state.plan_nodes):
            return ""
        if not self._reserve_solver():
            return None
        node = state.plan_nodes[state.node_i]
        user_msg = build_solver_user_message(
            self.problem,
            node.instruction,
            injected_prompt=state.pending_inject,
            step_index=state.step_counter,
            running_summary=state.running_summary,
            force_final=state.step_counter >= self.max_steps - 1,
        )
        used_inject = state.pending_inject
        state.pending_inject = ""
        state.messages.append({"role": "user", "content": user_msg})
        with _SOLVER_SEM:
            if self.solver_pool is not None:
                obs = self.solver_pool.chat(
                    state.messages,
                    temperature=self.agent.solver_temperature,
                    max_tokens=self.agent.solver_max_tokens,
                )
            else:
                obs = self.agent._solver_step(state.messages)
        state.messages.append({"role": "assistant", "content": obs})
        state.steps.append(
            StepRecord(
                index=state.step_counter,
                instruction=node.instruction,
                observation=obs,
                review=None,
                injected_prompt=used_inject,
                is_final=False,
                retried=state.retries_on_node > 0,
            )
        )
        state.step_counter += 1
        return obs

    def _apply_review(self, state: PathState, review: StepReview) -> str:
        state.steps[-1].review = review
        if review.summary:
            state.running_summary = review.summary
        action = review.action
        if action == "FINALIZE":
            return "leaf"
        if action == "RETRY" and state.retries_on_node < self.agent.max_retries_per_node:
            state.retries_on_node += 1
            state.pending_inject = review.hint
            return "continue"
        if action == "RETRY":
            return "leaf"
        if action == "REPLAN":
            return "replan"
        state.pending_inject = review.hint if action == "INJECT" else ""
        state.retries_on_node = 0
        state.node_i += 1
        if state.node_i >= len(state.plan_nodes):
            return "leaf"
        return "continue"

    def _act_on_review(
        self,
        rev_nid: str,
        st: PathState,
        review: StepReview,
        obs: str,
    ) -> None:
        mode = self._apply_review(st, review)
        if mode == "leaf":
            self._make_leaf(rev_nid, st, reason="finalize", observation=obs)
            return
        if mode == "replan":
            ctx = (
                f"Running summary:\n{st.running_summary}\n"
                f"Last issue:\n{review.issue}\n"
                f"Last observation (truncated):\n{obs[:800]}"
            )
            # Single replan (no k-fork) — keeps the tree binary per turn; the
            # review siblings already provide the DPO contrast at this state.
            self._branch_plans(rev_nid, st, context=ctx, pending_hint=review.hint, n=1)
            return
        self.expand_from(rev_nid, st)

    def expand_from(self, parent_nid: str | None, state: PathState) -> None:
        if not self._budget_ok():
            self._make_leaf(parent_nid, state, reason="cap_budget")
            return
        if state.step_counter >= self.max_steps:
            self._make_leaf(parent_nid, state, reason="cap_steps")
            return
        if state.node_i >= len(state.plan_nodes):
            self._make_leaf(parent_nid, state, reason="empty_plan")
            return

        obs = self._solver_once(state)
        if obs is None:
            self._make_leaf(parent_nid, state, reason="cap_budget")
            return
        if state.step_counter >= self.max_steps:
            # Final allowed turn: solver was forced to commit to an answer.
            self._make_leaf(parent_nid, state, reason="forced_final", observation=obs)
            return
        node = state.plan_nodes[min(state.node_i, len(state.plan_nodes) - 1)]
        is_last = state.node_i >= len(state.plan_nodes) - 1 or node.is_final
        turn_info = format_turn_info(state.step_counter, self.max_steps)
        # Clean prompt (no contrastive block) — this is what DPO trains on.
        train_prompt = build_review_user(
            self.problem,
            instruction=node.instruction,
            observation=obs,
            hist=format_history(state.steps[:-1]),
            running_summary=state.running_summary,
            inject_prior=node.inject_after,
            is_last_planned=is_last,
            turn_info=turn_info,
        )

        # Sample k sibling reviews SEQUENTIALLY so later ones can diverge from
        # earlier outputs (contrastive branching; collection-only prompt).
        branches: list[tuple[str, PathState, StepReview]] = []
        sibling_raws: list[str] = []
        for j in range(self.k):
            st = self._clone_state(state)
            review, raw = self.agent.sample_review(
                self.problem,
                instruction=node.instruction,
                observation=obs,
                history=st.steps[:-1],
                running_summary=st.running_summary,
                inject_prior=node.inject_after,
                is_last_planned=is_last,
                turn_info=turn_info,
                avoid_texts=sibling_raws if (self.contrastive and j > 0) else None,
            )
            sibling_raws.append(raw)
            rev_nid = self._add(
                kind="review",
                parent=parent_nid,
                orch_raw=raw,
                orch_parsed={
                    "summary": review.summary,
                    "status": review.status,
                    "issue": review.issue,
                    "hint": review.hint,
                    "action": review.action,
                },
                extra={
                    "instruction": node.instruction,
                    "observation": self._trunc(obs),
                    "step_index": st.step_counter - 1,
                    "running_summary_before": st.running_summary,
                    "parseable_answer": has_parseable_answer(obs),
                    "contrastive": bool(self.contrastive and j > 0),
                    "turn_info": turn_info,
                    "train_prompt": train_prompt,
                },
            )
            branches.append((rev_nid, st, review))

        # Expand subtrees in parallel (solver calls dominate).
        with ThreadPoolExecutor(max_workers=self.k) as pool:
            futs = [
                pool.submit(self._act_on_review, rev_nid, st, review, obs)
                for rev_nid, st, review in branches
            ]
            for fut in futs:
                fut.result()

    def _branch_plans(
        self,
        parent_nid: str | None,
        state: PathState | None = None,
        *,
        context: str = "",
        pending_hint: str = "",
        n: int | None = None,
    ) -> None:
        # Sequential contrastive plan sampling, then parallel subtree expansion.
        n = n if n is not None else self.k
        used_turns = state.step_counter if state is not None else 0
        turn_info = format_turn_info(used_turns, self.max_steps)
        train_prompt = build_plan_user(self.problem, context=context, turn_info=turn_info)
        branches: list[tuple[str, PathState]] = []
        sibling_raws: list[str] = []
        for j in range(n):
            if not self._budget_ok():
                break
            plan, raw = self.agent.sample_plan(
                self.problem,
                context=context,
                turn_info=turn_info,
                avoid_texts=sibling_raws if (self.contrastive and j > 0) else None,
            )
            sibling_raws.append(raw)
            plan_nid = self._add(
                kind="plan",
                parent=parent_nid,
                orch_raw=raw,
                orch_parsed={
                    "nodes": [
                        {
                            "instruction": n_.instruction,
                            "inject_after": n_.inject_after,
                            "is_final": n_.is_final,
                        }
                        for n_ in plan.nodes
                    ]
                },
                extra={
                    "context": context[:1000],
                    "contrastive": bool(self.contrastive and j > 0),
                    "turn_info": turn_info,
                    "train_prompt": train_prompt,
                },
            )
            if state is None:
                st = PathState(plan_nodes=list(plan.nodes))
            else:
                st = self._clone_state(state)
                st.plan_nodes = list(plan.nodes)
                st.node_i = 0
                st.retries_on_node = 0
                st.pending_inject = pending_hint
            branches.append((plan_nid, st))

        with ThreadPoolExecutor(max_workers=max(len(branches), 1)) as pool:
            futs = [
                pool.submit(self.expand_from, plan_nid, st)
                for plan_nid, st in branches
            ]
            for fut in futs:
                fut.result()

    def build(self) -> dict[str, Any]:
        root = self._add(
            kind="root", parent=None, extra={"problem_preview": self.problem[:200]}
        )
        self._branch_plans(root, None, context="")
        with self._lock:
            leaves = [n for n in self.nodes.values() if n["is_leaf"]]
            n_leaves = self.n_leaves
            n_solver = self.n_solver_calls
            truncated = self.truncated
            nodes = list(self.nodes.values())
        return {
            "nodes": nodes,
            "n_solver_calls": n_solver,
            "n_leaves": n_leaves,
            "n_orch_nodes": sum(1 for n in nodes if n["kind"] in {"plan", "review"}),
            "truncated": truncated,
            "leaf_em_mean": (
                sum(float(n["leaf_em"] or 0.0) for n in leaves) / n_leaves if n_leaves else 0.0
            ),
        }


def _load_done_ids(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") is not None and rec.get("error") is None:
                done.add(int(rec["id"]))
    return done


def _collect_one(
    row: dict,
    *,
    rr: _RoundRobin,
    solver_pool: _SolverPool | None,
    orch_url: str,
    orch_model: str,
    solver_model: str,
    k: int,
    max_turns: int,
    max_solver_calls: int,
    max_leaves: int,
    contrastive: bool,
    leaf_em_samples: int,
    leaf_resample_temp: float,
) -> dict:
    url = rr.next()
    t0 = time.time()
    try:
        agent = HintFlowAgent(
            orch_url=orch_url,
            orch_model=orch_model,
            solver_url=url,
            solver_model=solver_model,
            orch_temperature=0.8,
        )
        builder = TreeBuilder(
            agent=agent,
            problem=row["problem"],
            gold=row["gold"],
            problem_id=row["id"],
            k=k,
            max_steps=max_turns,
            max_solver_calls=max_solver_calls,
            max_leaves=max_leaves,
            contrastive=contrastive,
            leaf_em_samples=leaf_em_samples,
            leaf_resample_temp=leaf_resample_temp,
            solver_pool=solver_pool,
        )
        tree = builder.build()
        with _PRINT_LOCK:
            print(
                f"[done] id={row['id']} leaves={tree['n_leaves']} "
                f"solver={tree['n_solver_calls']} em_mean={tree['leaf_em_mean']:.2f} "
                f"sec={time.time()-t0:.1f}",
                flush=True,
            )
        return {
            "id": row["id"],
            "gold": row["gold"],
            "problem": row["problem"],
            "k": k,
            "solver_url": url,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": None,
            **tree,
        }
    except Exception as e:
        with _PRINT_LOCK:
            print(f"[err] id={row['id']}: {type(e).__name__}: {e}", flush=True)
        return {
            "id": row["id"],
            "gold": row.get("gold", ""),
            "problem": row.get("problem", ""),
            "k": k,
            "solver_url": url,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {e}",
            "nodes": [],
            "n_solver_calls": 0,
            "n_leaves": 0,
            "n_orch_nodes": 0,
            "truncated": False,
            "leaf_em_mean": 0.0,
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids-file", default=str(_HERE / "tree_collect_ids_2000.json"))
    p.add_argument("--data-file", default=str(_ROOT / "data" / "train.jsonl"))
    p.add_argument(
        "--out",
        default=str(_ROOT / "checkpoints" / "hintflow_trees_v2" / "trees.jsonl"),
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=24, help="parallel problems (default 24)")
    p.add_argument(
        "--max-inflight-solver",
        type=int,
        default=224,
        help="global cap on concurrent OSS calls; match n_servers * max-num-seqs",
    )
    p.add_argument(
        "--sticky-routing",
        action="store_true",
        help="pin each problem to one solver (old behavior); default is per-call RR",
    )
    p.add_argument("--k", type=int, default=2)
    p.add_argument(
        "--max-turns",
        type=int,
        default=7,
        help="hard tree depth (solver turns per path); last turn forces the answer",
    )
    p.add_argument(
        "--no-contrastive",
        action="store_true",
        help="disable contrastive sibling sampling (falls back to i.i.d. sampling)",
    )
    p.add_argument(
        "--leaf-em-samples",
        type=int,
        default=3,
        help="terminal answer sampled this many times total; leaf EM = mean",
    )
    p.add_argument("--leaf-resample-temp", type=float, default=0.7)
    p.add_argument(
        "--max-solver-calls",
        type=int,
        default=0,
        help="per-tree OSS cap; 0 = unlimited (no budget)",
    )
    p.add_argument(
        "--max-leaves",
        type=int,
        default=0,
        help="per-tree leaf cap; 0 = unlimited (no budget)",
    )
    p.add_argument("--solver-urls", default=",".join(DEFAULT_SOLVER_URLS))
    p.add_argument("--orch-url", default=ORCH_URL)
    p.add_argument("--orch-model", default=ORCH_MODEL)
    p.add_argument("--solver-model", default=SOLVER_MODEL)
    p.add_argument("--smoke", type=int, default=0)
    args = p.parse_args()

    ids = json.loads(Path(args.ids_file).read_text())
    if args.limit:
        ids = ids[: args.limit]
    if args.smoke:
        ids = ids[: args.smoke]

    by_id = {int(r["id"]): r for r in load_jsonl(Path(args.data_file))}
    rows = []
    for i in ids:
        r = by_id.get(int(i))
        if r is None:
            print(f"[warn] missing id {i}", flush=True)
            continue
        rows.append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_ids(out_path)
    rows = [r for r in rows if int(r["id"]) not in done]
    print(
        f"tree collect: todo={len(rows)} done={len(done)} workers={args.workers} "
        f"k={args.k} max_turns={args.max_turns} "
        f"max_solver={args.max_solver_calls} max_leaves={args.max_leaves} "
        f"routing={'sticky' if args.sticky_routing else 'per-call-rr'} "
        f"inflight={args.max_inflight_solver}",
        flush=True,
    )
    if not rows:
        print("nothing to do", flush=True)
        return

    global _SOLVER_SEM
    _SOLVER_SEM = threading.Semaphore(max(args.max_inflight_solver, 1))

    solver_urls = [u.strip() for u in args.solver_urls.split(",") if u.strip()]
    rr = _RoundRobin(solver_urls)
    solver_pool = None if args.sticky_routing else _SolverPool(solver_urls, args.solver_model)
    lock = threading.Lock()
    n_ok = n_err = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=min(args.workers, len(rows))) as pool:
        futs = {
            pool.submit(
                _collect_one,
                row,
                rr=rr,
                solver_pool=solver_pool,
                orch_url=args.orch_url,
                orch_model=args.orch_model,
                solver_model=args.solver_model,
                k=args.k,
                max_turns=args.max_turns,
                max_solver_calls=args.max_solver_calls,
                max_leaves=args.max_leaves,
                contrastive=not args.no_contrastive,
                leaf_em_samples=args.leaf_em_samples,
                leaf_resample_temp=args.leaf_resample_temp,
            ): row["id"]
            for row in rows
        }
        with out_path.open("a", encoding="utf-8") as fout:
            for fut in tqdm(as_completed(futs), total=len(futs), desc="trees"):
                rec = fut.result()
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    if rec.get("error"):
                        n_err += 1
                    else:
                        n_ok += 1

    elapsed = time.time() - t0
    summary = {
        "out": str(out_path),
        "n_ok": n_ok,
        "n_err": n_err,
        "elapsed_sec": round(elapsed, 1),
        "k": args.k,
        "max_turns": args.max_turns,
        "max_solver_calls": args.max_solver_calls,
        "max_leaves": args.max_leaves,
        "contrastive": not args.no_contrastive,
        "leaf_em_samples": args.leaf_em_samples,
        "leaf_resample_temp": args.leaf_resample_temp,
        "orch_model_dir": "hintflow_dpo_merged (served as qwen3-4b @8086)",
    }
    summary_path = out_path.with_suffix(".summary.json")
    prev: dict = {}
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text())
        except Exception:
            prev = {}
    summary["n_ok_total"] = int(prev.get("n_ok_total", 0)) + n_ok
    summary["n_err_total"] = int(prev.get("n_err_total", 0)) + n_err
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
