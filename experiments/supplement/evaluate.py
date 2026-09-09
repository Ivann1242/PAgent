"""Matched HF1 candidate generation, offline selector swaps, capped HFM."""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import time

from core import exact_match, extract_final_answer
from HintFlow_one.one_agent import HintFlowOneAgent, Candidate, SELECTOR_SYSTEM, DEFAULT_FIXED_COT, _message_text
from HintFlow_draft.agent import DraftStepAgent

from .io import dataset, digest, write_json
from .trace import Ledger, TracedClient


@lru_cache(maxsize=2)
def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def trial_seed(qid, seed):
    return int(digest([str(qid), seed])[:8], 16) % (2**30)


def cache(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text())
    # An interrupted request is not silently retried: its state/cost may be
    # unknown. Human inspection is required before clearing a .inflight file.
    marker = path.with_suffix(".inflight")
    if marker.exists():
        raise RuntimeError(f"Interrupted trial needs audit before retry: {marker}")
    write_json(marker, {"started": time.time()})
    result = fn()
    write_json(path, result)
    marker.unlink()
    return result


class MatchedAgent(HintFlowOneAgent):
    def __init__(self, cfg, path, seed, *, mode="blind_ff", cap=None):
        super().__init__(
            orch_url=f"http://127.0.0.1:{cfg['policy_port']}/v1", orch_model="current",
            solver_url=f"http://127.0.0.1:{cfg['solver_port']}/v1", solver_model="qwen3-14b",
            solver_max_tokens=cap or cfg["solver_max_tokens"],
            solver_temperature=cfg["solver_temperature"],
            challenger_temperature=cfg["challenger_temperature"], challenger_mode=mode,
            replace_threshold=cfg["replace_threshold"], solver_seed=seed,
            request_timeout=cfg["request_timeout"],
        )
        self.select_model = "sft"
        self.orch = TracedClient(self.orch, path, role="policy")
        self.solver = TracedClient(self.solver, path, role="solver")

    def _orch_chat(self, user, *, system=None, max_tokens=192):
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        response = self.orch.chat.completions.create(
            model=self.select_model if system == SELECTOR_SYSTEM else self.orch_model,
            messages=messages, temperature=self.orch_temperature, max_tokens=max_tokens,
            seed=self.solver_seed, top_p=1.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _message_text(response)

    def close(self):
        self.orch.close()
        self.solver.close()


def evaluate_hf1(cfg, condition, train_seed, *, cap=None, budget_tag=None):
    root = Path(cfg["output"]) / (budget_tag or "eval")
    cap = cap or cfg["solver_max_tokens"]
    modes = ["blind_ff"] if budget_tag or condition != "sft" else ["resample_baseline", "fixed_cot", "blind_ff"]
    for row in dataset(cfg["eval_data"]):
        for seed in cfg["eval_seeds"]:
            key = digest([row["id"], seed])[:24]
            base_path = root / "bare" / f"{key}.json"

            def make_bare():
                agent = MatchedAgent(cfg, base_path.with_suffix(".calls.jsonl"), trial_seed(row["id"], seed), cap=cap)
                try:
                    return {"candidate": asdict(agent.generate_baseline(row["problem"])), "error": None}
                except Exception as exc:
                    return {"candidate": None, "error": str(exc)}
                finally:
                    agent.close()

            baseline = cache(base_path, make_bare)
            for mode in modes:
                method = {"resample_baseline": "resample", "fixed_cot": "fixed_cot"}.get(mode, condition)
                path = root / "candidates" / method / str(train_seed) / f"{key}.json"

                def make_candidate():
                    agent = MatchedAgent(cfg, path.with_suffix(".calls.jsonl"), trial_seed(row["id"], seed), mode=mode, cap=cap)
                    # Cached bare still occupies conceptual solver call 0.
                    agent._solver_call_index = 1
                    partial = {"candidate": None, "hint": "", "hint_raw": "", "error": None}
                    try:
                        if baseline["error"]:
                            raise RuntimeError("shared baseline failed: " + baseline["error"])
                        if mode == "blind_ff":
                            hint, raw, ok = agent.generate_hint(row["problem"])
                        elif mode == "fixed_cot":
                            hint, raw, ok = DEFAULT_FIXED_COT, DEFAULT_FIXED_COT, True
                        else:
                            hint, raw, ok = "", "", True
                        partial.update(hint=hint, hint_raw=raw, hint_parse_ok=ok,
                                       hint_tokens=len(tokenizer(cfg["policy_base"])(hint, add_special_tokens=False)["input_ids"]))
                        # Persist hint even if solver subsequently fails.
                        write_json(path.with_suffix(".partial.json"), partial)
                        partial["candidate"] = asdict(agent.generate_challenger(row["problem"], hint))
                    except Exception as exc:
                        partial["error"] = str(exc)
                    finally:
                        agent.close()
                    return partial

                challenger = cache(path, make_candidate)
                selectors = ["sft", "current"] if method not in {"sft", "resample", "fixed_cot"} else ["sft"]
                for selector in selectors:
                    result_path = root / "results" / method / str(train_seed) / selector / f"{key}.json"

                    def select():
                        base = Candidate(**baseline["candidate"]) if baseline["candidate"] else None
                        chal = Candidate(**challenger["candidate"]) if challenger["candidate"] else None
                        base_em = int(exact_match(base.answer, str(row["gold"]))) if base and base.answer else 0
                        chal_em = int(exact_match(chal.answer, str(row["gold"]))) if chal and chal.answer else 0
                        selected = base
                        decision = None
                        if base and chal:
                            agent = MatchedAgent(cfg, result_path.with_suffix(".calls.jsonl"), trial_seed(row["id"], seed), mode=mode, cap=cap)
                            agent.select_model = selector
                            try:
                                decision = asdict(agent.select(row["problem"], base, chal))
                                if decision["decision"] == "REPLACE" and decision["confidence"] >= cfg["replace_threshold"]:
                                    selected = chal
                            finally:
                                agent.close()
                        final = int(exact_match(selected.answer, str(row["gold"]))) if selected and selected.answer else 0
                        return {"id": row["id"], "eval_seed": seed, "train_seed": train_seed,
                                "method": method, "selector": selector, "em": final,
                                "baseline_em": base_em, "challenger_em": chal_em,
                                "union": max(base_em, chal_em), "random_expected_em": (base_em + chal_em) / 2,
                                "recovered": int(not base_em and final), "harmed": int(base_em and not final),
                                "selection": decision, "error": baseline["error"] or challenger["error"],
                                "baseline_missing": base is None, "challenger_missing": chal is None,
                                "baseline_source": str(base_path), "candidate_source": str(path),
                                "solver_output_cap": 2 * cap, "final_answer": selected.answer if selected else ""}

                    cache(result_path, select)
                # Swap the current selector into the fixed SFT candidate pool,
                # without generating the SFT candidates again.
                if method == "vp" and not budget_tag:
                    swap_selector(cfg, root, row, seed, train_seed, key, baseline)


def swap_selector(cfg, root, row, seed, train_seed, key, baseline):
    sft_seed = cfg["seeds"][0]
    source = root / "candidates" / "sft" / str(sft_seed) / f"{key}.json"
    if not source.exists():
        raise RuntimeError("Evaluate SFT candidates before selector exchange")
    challenger = json.loads(source.read_text())
    result_path = root / "results" / "sft_swap" / str(train_seed) / "current" / f"{key}.json"

    def select():
        base = Candidate(**baseline["candidate"]) if baseline["candidate"] else None
        chal = Candidate(**challenger["candidate"]) if challenger["candidate"] else None
        chosen = base
        decision = None
        if base and chal:
            agent = MatchedAgent(cfg, result_path.with_suffix(".calls.jsonl"), trial_seed(row["id"], seed))
            agent.select_model = "current"
            try:
                decision = asdict(agent.select(row["problem"], base, chal))
                if decision["decision"] == "REPLACE" and decision["confidence"] >= cfg["replace_threshold"]:
                    chosen = chal
            finally:
                agent.close()
        grade = lambda c: int(exact_match(c.answer, str(row["gold"]))) if c and c.answer else 0
        b, c, f = grade(base), grade(chal), grade(chosen)
        return {"id": row["id"], "eval_seed": seed, "train_seed": train_seed,
                "method": "sft_swap", "selector": "current", "baseline_em": b,
                "challenger_em": c, "em": f, "union": max(b, c), "random_expected_em": (b+c)/2,
                "recovered": int(not b and f), "harmed": int(b and not f), "selection": decision,
                "error": baseline["error"] or challenger["error"], "baseline_missing": base is None,
                "challenger_missing": chal is None, "baseline_source": str(root / "bare" / f"{key}.json"),
                "candidate_source": str(source), "final_answer": chosen.answer if chosen else ""}
    cache(result_path, select)


class CappedDraft(DraftStepAgent):
    def __init__(self, cfg, path, seed, budget):
        super().__init__(small_url=f"http://127.0.0.1:{cfg['policy_port']}/v1", small_model="current",
                         solver_url=f"http://127.0.0.1:{cfg['solver_port']}/v1", solver_model="qwen3-14b",
                         max_steps=cfg["max_steps_hfm"], draft_max_tokens=budget // 4,
                         temperature=cfg["solver_temperature"], seed=seed,
                         request_timeout=cfg["request_timeout"])
        self.budget = budget
        self.ledger = Ledger(budget)
        self.draft_text = None
        self.solver = TracedClient(self.solver, path, role="solver", ledger=self.ledger)
        self.planner = TracedClient(self.planner, path, role="planner")
        self.hinter = TracedClient(self.hinter, path, role="hinter")
        self.selector = TracedClient(self.selector, path, role="selector")

    def generate_draft(self, problem):
        prompt, text = super().generate_draft(problem)
        self.draft_text = text
        return prompt, text

    def segment_draft(self, problem, draft):
        result = super().segment_draft(problem, draft)
        self.step_max_tokens = (self.budget - self.draft_max_tokens) // (2 * len(result[0]))
        return result

    def close(self):
        for client in (self.solver, self.planner, self.hinter, self.selector):
            client.close()


def evaluate_hfm(cfg, condition, train_seed):
    for budget in cfg["budgets"]:
        tag = f"budget-{budget}"
        # Identical challenger/bare temperature for the budget comparison.
        matched = {**cfg, "challenger_temperature": cfg["solver_temperature"]}
        evaluate_hf1(matched, condition, train_seed, cap=budget // 2, budget_tag=tag)
        root = Path(cfg["output"]) / tag / "hfm"
        for row in dataset(cfg["eval_data"]):
            for seed in cfg["eval_seeds"]:
                key = digest([row["id"], seed])[:24]
                path = root / f"{key}.json"

                def trial():
                    agent = CappedDraft(cfg, path.with_suffix(".calls.jsonl"), trial_seed(row["id"], seed), budget)
                    try:
                        result = agent.run(row["problem"], gold="").to_dict()
                        result["em"] = int(exact_match(result["final_answer"], str(row["gold"]))) if result["final_answer"] else 0
                    except Exception as exc:
                        result = {"em": 0, "error": str(exc), "draft": agent.draft_text}
                    finally:
                        agent.close()
                    draft = agent.draft_text
                    answer = extract_final_answer(draft) if draft else ""
                    result.update(id=row["id"], eval_seed=seed, train_seed=train_seed,
                                  method="hfm", budget=budget, reserved_cap=agent.ledger.reserved,
                                  draft_missing=draft is None,
                                  draft_em=(int(exact_match(answer, str(row["gold"]))) if answer else 0) if draft is not None else None)
                    return result

                cache(path, trial)
