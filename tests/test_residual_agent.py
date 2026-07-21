import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HINTFLOW = ROOT / "HintFlow"
for path in (ROOT, HINTFLOW):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_turn_oracle import summarize_residual
from core import exact_match, extract_final_answer
from export_residual_feedback import export_counterfactual
from residual_agent import (
    Candidate,
    ResidualHintFlowAgent,
    Selection,
    TurnFeedback,
    build_residual_solver_prompt,
)


class FakeResidualAgent(ResidualHintFlowAgent):
    def __init__(self, outputs, decisions, **kwargs):
        super().__init__(
            orch_url="http://unused",
            solver_url="http://unused",
            **kwargs,
        )
        self.outputs = iter(outputs)
        self.decisions = iter(decisions)

    def _solver_chat(self, prompt, *, seed=None, temperature=0.0, max_tokens=None):
        del prompt, seed, temperature, max_tokens
        return next(self.outputs)

    def feedback(self, problem, candidate, *, tried_actions=None, remaining_calls=0):
        del problem, candidate, tried_actions, remaining_calls
        return TurnFeedback(
            p_correct=0.5,
            error_type="UNKNOWN",
            action="VERIFY_REPAIR",
        )

    def select(self, problem, incumbent, challenger):
        del problem, incumbent, challenger
        return next(self.decisions)


class ResidualAgentTest(unittest.TestCase):
    def test_math_exact_match_preserves_sign_and_fraction_structure(self):
        self.assertEqual(exact_match("-2", "2"), 0)
        self.assertEqual(exact_match("1/2", "12"), 0)
        self.assertEqual(exact_match("0.5", "1/2"), 1)
        self.assertEqual(exact_match(r"\frac{1}{2}", "0.5"), 1)
        self.assertEqual(exact_match("1,2", "12"), 0)
        self.assertEqual(exact_match("1,000", "1000"), 1)

    def test_final_answer_extraction_handles_markdown_and_latex_wrappers(self):
        self.assertEqual(extract_final_answer("**Final Answer:** 16"), "16")
        self.assertEqual(extract_final_answer(r"Final Answer: \boxed{16}"), "16")
        self.assertEqual(
            extract_final_answer("Final Answer:\n\\[16\\]"),
            "16",
        )

    def test_correct_baseline_is_retained_against_bad_challenger(self):
        agent = FakeResidualAgent(
            ["Final Answer: 7", "Final Answer: 9"],
            [Selection("KEEP", 1.0)],
            max_solver_calls=2,
            action_schedule=("VERIFY_REPAIR",),
        )
        trajectory = agent.run("dummy", gold="7")
        self.assertEqual(trajectory.baseline_em, 1)
        self.assertEqual(trajectory.em, 1)
        self.assertEqual(trajectory.incumbent_index, 0)
        self.assertEqual(len(trajectory.candidates), 2)

    def test_wrong_baseline_can_be_replaced(self):
        agent = FakeResidualAgent(
            ["Final Answer: 9", "Final Answer: 7"],
            [Selection("REPLACE", 0.99)],
            max_solver_calls=2,
            replace_threshold=0.7,
            action_schedule=("ALTERNATE_SOLVE",),
        )
        trajectory = agent.run("dummy", gold="7")
        self.assertEqual(trajectory.baseline_em, 0)
        self.assertEqual(trajectory.em, 1)
        self.assertEqual(trajectory.oracle_em, 1)
        self.assertEqual(trajectory.incumbent_index, 1)

    def test_budget_caps_solver_candidates(self):
        outputs = [f"Final Answer: {i}" for i in range(7)]
        decisions = [Selection("KEEP", 1.0) for _ in range(6)]
        agent = FakeResidualAgent(
            outputs,
            decisions,
            max_solver_calls=7,
        )
        trajectory = agent.run("dummy", gold="99")
        self.assertEqual(len(trajectory.candidates), 7)
        self.assertEqual(len(trajectory.turns), 6)
        with self.assertRaises(ValueError):
            FakeResidualAgent([], [], max_solver_calls=8)

    def test_alternate_prompt_avoids_incumbent_anchoring(self):
        prompt = build_residual_solver_prompt(
            "problem",
            "ALTERNATE_SOLVE",
            incumbent="SECRET INCUMBENT",
            feedback=TurnFeedback(),
        )
        self.assertNotIn("SECRET INCUMBENT", prompt)

    def test_summary_reports_paired_recovery(self):
        agent = FakeResidualAgent(
            ["Final Answer: 9", "Final Answer: 7"],
            [Selection("REPLACE", 1.0)],
            max_solver_calls=2,
            action_schedule=("ALTERNATE_SOLVE",),
        )
        record = agent.run("dummy", gold="7").to_dict()
        record.update({"id": 1, "error": None})
        summary = summarize_residual([record], bootstrap_samples=20)
        self.assertEqual(summary["recovered_wrong_to_right"], 1)
        self.assertEqual(summary["harmed_right_to_wrong"], 0)
        self.assertEqual(summary["paired_delta"], 1)

    def test_counterfactual_export_separates_short_label_tasks(self):
        record = {
            "id": 1,
            "problem_id": 1,
            "problem": "dummy",
            "gold": "7",
            "incumbent": {
                "action": "BASELINE",
                "solution": "Final Answer: 9",
                "em": 0,
            },
            "policy_feedback": {
                "p_correct": 0.1,
                "error_type": "ARITHMETIC",
                "evidence": "bad sum",
            },
            "teacher_feedback": {
                "error_type": "ARITHMETIC",
                "evidence": "bad sum",
                "repair_hint": "recompute",
            },
            "branches": [
                {
                    "action": "VERIFY_REPAIR",
                    "candidate": {
                        "action": "VERIFY_REPAIR",
                        "solution": "Final Answer: 7",
                        "em": 1,
                    },
                }
            ],
            "action_values": {
                "STOP": {"q": 0.0, "n": 1},
                "VERIFY_REPAIR": {"q": 1.0, "n": 1},
            },
            "baseline_em": 0,
            "error": None,
        }
        rows = export_counterfactual([record], action_cost=0.02)
        targets = {(row["task"], row["target"]) for row in rows}
        self.assertIn(("correctness", "CORRECT"), targets)
        self.assertIn(("correctness", "INCORRECT"), targets)
        self.assertIn(("selection", "REPLACE"), targets)
        self.assertIn(("action", "VERIFY_REPAIR"), targets)
        self.assertTrue(any(row["task"] == "diagnosis" for row in rows))


if __name__ == "__main__":
    unittest.main()
