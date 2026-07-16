import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HINTFLOW = ROOT / "HintFlow"
for path in (ROOT, HINTFLOW):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from HintFlowAgent import (
    CandidateSelection,
    HintFlowAgent,
    Plan,
    PlanNode,
    StepReview,
    _message_text,
    build_compact_solver_message,
    parse_candidate_selection,
    parse_step_result,
)
from core import build_large_prompt


class FakeHintFlowAgent(HintFlowAgent):
    def __init__(self, outputs, reviews, selections=None, **kwargs):
        super().__init__(
            orch_url="http://unused",
            solver_url="http://unused",
            **kwargs,
        )
        self.outputs = iter(outputs)
        self.reviews = iter(reviews)
        self.selection_outputs = iter(selections or [])
        self.solver_message_batches = []

    def _solver_step(self, messages):
        self.solver_message_batches.append(messages)
        return next(self.outputs)

    def review_and_control(self, problem, **kwargs):
        del problem, kwargs
        return next(self.reviews)

    def _select_candidate(self, problem, incumbent, challenger):
        del problem, incumbent, challenger
        return next(self.selection_outputs)


def one_step_plan():
    return Plan(
        nodes=[
            PlanNode(
                "Solve and return the requested answer.",
                is_final=True,
            )
        ]
    )


def two_step_plan():
    return Plan(
        nodes=[
            PlanNode("Solve independently.", is_final=False),
            PlanNode("Verify and return the answer.", is_final=True),
        ]
    )


class HintFlowAgentV2Test(unittest.TestCase):
    def test_structured_step_result_parses_compact_state(self):
        result = parse_step_result(
            '{"result":"x=2","key_equations":["x+1=3"],'
            '"candidate_answer":"2","is_complete":true,'
            '"confidence":"high","uncertainty":""}'
        )
        self.assertTrue(result.parse_ok)
        self.assertTrue(result.is_complete)
        self.assertEqual(result.candidate_answer, "2")
        self.assertIn("x+1=3", result.compact_text())

    def test_malformed_step_result_is_not_promoted(self):
        result = parse_step_result("not json")
        self.assertFalse(result.parse_ok)
        self.assertFalse(result.is_complete)
        self.assertEqual(result.candidate_answer, "")

    def test_malformed_step_result_preserves_explicit_final_answer(self):
        result = parse_step_result("work was truncated\nFinal Answer: 7")
        self.assertFalse(result.parse_ok)
        self.assertTrue(result.is_complete)
        self.assertEqual(result.candidate_answer, "7")

    def test_message_text_reads_gpt_oss_reasoning_field(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        reasoning="Final Answer: 7",
                        reasoning_content="",
                    )
                )
            ]
        )
        self.assertEqual(_message_text(response), "Final Answer: 7")

    def test_string_false_does_not_mark_result_complete(self):
        result = parse_step_result(
            '{"result":"draft","candidate_answer":"7","is_complete":"false"}'
        )
        self.assertFalse(result.is_complete)

    def test_non_object_selector_output_conservatively_keeps(self):
        selection = parse_candidate_selection("[]")
        self.assertEqual(selection.decision, "KEEP")
        self.assertEqual(selection.confidence, 0.0)

    def test_compact_prompt_marks_hypotheses_as_unverified(self):
        prompt = build_compact_solver_message(
            "p",
            "goal",
            accepted_state=[],
            candidate_state=[],
            step_index=0,
        )
        self.assertIn("[Current goal", prompt)
        self.assertNotIn("[Accepted results", prompt)
        self.assertIn('"candidate_answer"', prompt)

    def test_fresh_mode_never_sends_solver_history(self):
        agent = FakeHintFlowAgent(
            ["partial", "Final Answer: 7"],
            [
                StepReview(status="incomplete", action="RETRY", hint="finish"),
                StepReview(status="correct", action="FINALIZE"),
            ],
            runtime_mode="fresh",
            max_steps=2,
            max_retries_total=1,
        )
        trajectory = agent.reason("dummy", one_step_plan(), gold="7")
        self.assertEqual([len(batch) for batch in agent.solver_message_batches], [1, 1])
        self.assertEqual(trajectory.em, 1)
        self.assertEqual(trajectory.retry_count, 1)

    def test_retained_mode_preserves_correct_baseline(self):
        structured_wrong = (
            '{"result":"computed another value","key_equations":[],"candidate_answer":"9",'
            '"is_complete":true,"confidence":"high","uncertainty":""}'
        )
        agent = FakeHintFlowAgent(
            ["Final Answer: 7", structured_wrong],
            [StepReview(status="correct", action="FINALIZE")],
            [CandidateSelection("KEEP", 0.95, "incumbent is safer")],
            runtime_mode="retained",
            max_steps=2,
        )
        trajectory = agent.reason("dummy", one_step_plan(), gold="7")
        self.assertEqual(trajectory.baseline_em, 1)
        self.assertEqual(trajectory.em, 1)
        self.assertEqual(trajectory.incumbent_index, 0)
        self.assertEqual(len(trajectory.candidates), 2)
        self.assertIn("consensus guard", trajectory.selections[0].reason)

    def test_retained_mode_replaces_unparseable_baseline(self):
        structured_right = (
            '{"result":"checked result","key_equations":[],"candidate_answer":"7",'
            '"is_complete":true,"confidence":"high","uncertainty":""}'
        )
        agent = FakeHintFlowAgent(
            ["", structured_right],
            [StepReview(status="correct", action="FINALIZE")],
            runtime_mode="retained",
            max_steps=2,
        )
        trajectory = agent.reason("dummy", one_step_plan(), gold="7")
        self.assertEqual(trajectory.baseline_em, 0)
        self.assertEqual(trajectory.em, 1)
        self.assertEqual(trajectory.incumbent_index, 1)
        self.assertEqual(
            agent.solver_message_batches[0][0]["content"],
            build_large_prompt("dummy", ""),
        )

    def test_two_agreeing_challengers_can_replace_parseable_baseline(self):
        structured_right = (
            '{"result":"checked result","key_equations":[],"candidate_answer":"7",'
            '"is_complete":true,"confidence":"high","uncertainty":""}'
        )
        agent = FakeHintFlowAgent(
            ["Final Answer: 9", structured_right, structured_right],
            [
                StepReview(status="correct", action="NO_HINT"),
                StepReview(status="correct", action="FINALIZE"),
            ],
            [CandidateSelection("REPLACE", 0.95, "two challengers agree")],
            runtime_mode="retained",
            max_steps=3,
        )
        trajectory = agent.reason("dummy", two_step_plan(), gold="7")
        self.assertEqual(trajectory.baseline_em, 0)
        self.assertEqual(trajectory.em, 1)
        self.assertEqual(trajectory.incumbent_index, 2)

    def test_incomplete_structured_answer_is_archived_but_not_finalized(self):
        incomplete = (
            '{"result":"unverified draft","key_equations":[],"candidate_answer":"7",'
            '"is_complete":false,"confidence":"low","uncertainty":"not checked"}'
        )
        agent = FakeHintFlowAgent(
            [incomplete],
            [StepReview(status="incomplete", action="FINALIZE")],
            runtime_mode="structured",
            max_steps=1,
        )
        trajectory = agent.reason("dummy", one_step_plan(), gold="7")
        self.assertEqual(len(trajectory.candidates), 1)
        self.assertIsNone(trajectory.incumbent_index)
        self.assertEqual(trajectory.final_answer, "")
        self.assertEqual(trajectory.em, 0)


if __name__ == "__main__":
    unittest.main()
