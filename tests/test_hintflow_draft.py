import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HintFlow_draft.agent import DraftStepAgent
from HintFlow_draft.common import (
    AcceptedStep,
    StepPlan,
    blind_candidates,
    build_goal_step_prompt,
    build_segment_prompt,
    build_selector_prompt,
    build_step_hint_prompt,
    context_text,
    fallback_plan,
    parse_selector_output,
    parse_step_plan,
    prefer_draft_consistent,
    resolve_blinded_selection,
    sanitize_stub_leaked_plan,
)


VALID_PLAN = """{
  "steps": [
    {"goal": "Set up the governing equation.", "is_final": false},
    {"goal": "Verify the work and output Final Answer: <answer>.", "is_final": true}
  ]
}"""


class FakeDraftStepAgent(DraftStepAgent):
    def __init__(self):
        super().__init__(
            small_url="http://unused",
            small_model="small",
            solver_url="http://unused",
            solver_model="large",
            seed=17,
        )
        self.accepted_snapshots = []
        self.hint_calls = 0
        self.select_calls = 0

    def generate_draft(self, problem):
        del problem
        return "draft prompt", "Incorrect draft.\nFinal Answer: 99"

    def segment_draft(self, problem, draft):
        del problem, draft
        return (
            [
                StepPlan("Set up the equation.", False),
                StepPlan("Verify and output Final Answer: <answer>.", True),
            ],
            VALID_PLAN,
            True,
            "",
        )

    def generate_hint(self, problem, accepted, goal):
        del problem, goal
        self.hint_calls += 1
        self.accepted_snapshots.append(list(accepted))
        return "hint prompt", "use a check", "use a check", True

    def generate_step_candidate(
        self, problem, accepted, step, *, hint="", source
    ):
        del problem, hint
        self.accepted_snapshots.append(list(accepted))
        if step.is_final:
            answer = "7" if source == "HINTED" else "9"
            return f"{source} final prompt", f"Checked.\nFinal Answer: {answer}"
        return f"{source} step prompt", f"{source.lower()} equation result"

    def select_step(
        self,
        problem,
        accepted,
        step,
        baseline,
        hinted,
        *,
        step_index,
        draft="",
        plan_goals=None,
    ):
        del problem, accepted, step, baseline, hinted, step_index, draft, plan_goals
        self.select_calls += 1
        return (
            "HINTED",
            0.9,
            "better goal completion",
            True,
            "A",
            '{"decision":"A","confidence":0.9}',
            {"A": "HINTED", "B": "BASELINE"},
            "selector prompt",
        )


class InvalidSelectorAgent(DraftStepAgent):
    def __init__(self):
        super().__init__(
            small_url="http://unused",
            small_model="small",
            solver_url="http://unused",
            solver_model="large",
            seed=5,
        )

    def _call(self, *args, **kwargs):
        del args, kwargs
        return "not json"


class HintFlowDraftCommonTest(unittest.TestCase):
    def test_parse_plan_accepts_fences_and_json_invalid_latex_backslash(self):
        raw = r"""```json
{"steps":[
  {"goal":"Apply \gcd to the coefficients.","is_final":false},
  {"goal":"Verify and output Final Answer: <answer>.","is_final":true}
]}
```"""
        steps = parse_step_plan(raw)
        self.assertEqual(len(steps), 2)
        self.assertIn("\\gcd", steps[0].goal)
        self.assertTrue(steps[-1].is_final)

    def test_empty_plan_raises_and_has_single_step_fallback(self):
        with self.assertRaises(ValueError):
            parse_step_plan('{"steps":[]}')
        fallback = fallback_plan()
        self.assertEqual(len(fallback), 1)
        self.assertTrue(fallback[0].is_final)

    def test_plan_enforces_step_limit_and_unique_last_final(self):
        with self.assertRaises(ValueError):
            parse_step_plan(VALID_PLAN, max_steps=1)
        with self.assertRaises(ValueError):
            parse_step_plan(
                '{"steps":[{"goal":"x","is_final":true},'
                '{"goal":"y","is_final":true}]}'
            )
        with self.assertRaises(ValueError):
            parse_step_plan(
                '{"steps":[{"goal":"x","is_final":false}]}'
            )

    def test_context_pairs_goal_and_selected_result(self):
        rendered = context_text(
            [
                AcceptedStep("derive an equation", "x+1=3"),
                AcceptedStep("solve it", "x=2"),
            ]
        )
        self.assertIn("Goal: derive an equation", rendered)
        self.assertIn("Result:\nx=2", rendered)
        self.assertLess(rendered.index("x+1=3"), rendered.index("Goal: solve it"))

    def test_post_plan_prompts_do_not_receive_draft(self):
        secret = "DRAFT_SECRET_ANSWER_314159"
        accepted = [AcceptedStep("set up", "equation")]
        prompts = [
            build_step_hint_prompt("problem", accepted, "next goal"),
            build_goal_step_prompt(
                "problem", accepted, "next goal", hint="method", is_final=False
            ),
            build_selector_prompt(
                "problem", accepted, "next goal", "candidate one", "candidate two"
            ),
        ]
        self.assertTrue(all(secret not in prompt for prompt in prompts))

    def test_blinded_mapping_round_trip(self):
        candidate_a, candidate_b, mapping = blind_candidates(
            "bare", "guided", seed=11, step_index=2
        )
        self.assertEqual({candidate_a, candidate_b}, {"bare", "guided"})
        self.assertEqual(set(mapping.values()), {"BASELINE", "HINTED"})
        for label in ("A", "B"):
            self.assertEqual(resolve_blinded_selection(label, mapping), mapping[label])

    def test_segment_prompt_forbids_copying_draft_values(self):
        prompt = build_segment_prompt("problem", "draft with n=1 and Final Answer: 4")
        self.assertIn("Do not name specific candidate values", prompt)
        self.assertIn("the placeholder Final Answer: <answer>", prompt)

    def test_prefer_draft_consistent_picks_matching_last_step(self):
        choice = prefer_draft_consistent(
            "Baseline writeup.\nFinal Answer: 736",
            "Hinted writeup.\nFinal Answer: 371",
            "Draft writeup.\nFinal Answer: 371",
            is_final=True,
        )
        self.assertEqual(choice[0], "HINTED")
        self.assertIsNone(
            prefer_draft_consistent(
                "Baseline writeup.\nFinal Answer: 736",
                "Hinted writeup.\nFinal Answer: 371",
                "Draft writeup.\nFinal Answer: 371",
                is_final=False,
            )
        )
        self.assertIsNone(
            prefer_draft_consistent(
                "Baseline writeup.\nFinal Answer: 5",
                "Hinted writeup.\nFinal Answer: 5",
                "Draft writeup.\nFinal Answer: 4",
                is_final=True,
            )
        )
        self.assertIsNone(
            prefer_draft_consistent(
                "Baseline writeup.\nFinal Answer: 736",
                "Hinted writeup.\nFinal Answer: 371",
                "Draft writeup.\nFinal Answer: 10",
                is_final=True,
            )
        )
        self.assertIsNone(
            prefer_draft_consistent(
                "Baseline writeup.\nFinal Answer: 83592",
                "Hinted writeup.\nFinal Answer: 152",
                "Draft writeup.\nFinal Answer: 83592",
                is_final=True,
                goal="verify the work and produce Final Answer: 83592",
                plan_goals=[
                    "result-free executable goal",
                    "verify the work and produce Final Answer: 83592",
                ],
            )
        )
        self.assertEqual(
            prefer_draft_consistent(
                "Baseline writeup.\nFinal Answer: 55",
                "Hinted writeup.\nFinal Answer: 117",
                "Draft writeup.\nFinal Answer: 55",
                is_final=True,
                goal="Verify the work and produce Final Answer: 55",
                plan_goals=[
                    "Compute bigness for small integer values",
                    "Check consecutive values",
                    "Identify the smallest N",
                    "Verify the work and produce Final Answer: 55",
                ],
            )[0],
            "BASELINE",
        )

    def test_sanitize_stub_plan_strips_leaked_last_goal(self):
        cleaned, changed = sanitize_stub_leaked_plan(
            [
                StepPlan("result-free executable goal", False),
                StepPlan("verify the work and produce Final Answer: 83592", True),
            ]
        )
        self.assertTrue(changed)
        self.assertIn("Final Answer: <answer>", cleaned[-1].goal)
        self.assertNotIn("83592", cleaned[-1].goal)
        untouched, flag = sanitize_stub_leaked_plan(
            [
                StepPlan("Compute bigness for small values", False),
                StepPlan("Verify the work and produce Final Answer: 55", True),
            ]
        )
        self.assertFalse(flag)
        self.assertIn("55", untouched[-1].goal)

    def test_explicit_selector_tie_resolves_to_blinded_a(self):
        decision, confidence, reason = parse_selector_output(
            '{"decision":"A | B","confidence":0.7,"reason":"equivalent"}'
        )
        self.assertEqual(decision, "A")
        self.assertEqual(confidence, 0.7)
        self.assertEqual(reason, "equivalent")


class HintFlowDraftAgentTest(unittest.TestCase):
    def test_full_mode_uses_selected_trajectory_not_draft(self):
        agent = FakeDraftStepAgent()
        trajectory = agent.run("dummy", gold="7", mode="full")
        self.assertEqual(trajectory.draft_answer, "99")
        self.assertEqual(trajectory.draft_em, 0)
        self.assertEqual(trajectory.final_answer, "7")
        self.assertEqual(trajectory.em, 1)
        self.assertEqual(trajectory.recovered, 1)
        self.assertEqual(agent.hint_calls, 2)
        self.assertEqual(agent.select_calls, 2)
        second_step_snapshots = [
            snapshot for snapshot in agent.accepted_snapshots if len(snapshot) == 1
        ]
        self.assertTrue(second_step_snapshots)
        self.assertEqual(
            second_step_snapshots[0][0].result, "hinted equation result"
        )

    def test_baseline_mode_skips_hint_and_selector(self):
        agent = FakeDraftStepAgent()
        trajectory = agent.run("dummy", gold="9", mode="baseline")
        self.assertEqual(trajectory.final_answer, "9")
        self.assertEqual(agent.hint_calls, 0)
        self.assertEqual(agent.select_calls, 0)
        self.assertTrue(all(step.decision == "BASELINE" for step in trajectory.steps))

    def test_selector_parse_failure_fails_closed_to_baseline(self):
        agent = InvalidSelectorAgent()
        result = agent.select_step(
            "problem",
            [],
            StepPlan("current goal", False),
            "baseline result",
            "hinted result",
            step_index=1,
        )
        decision, confidence, reason, parse_ok = result[:4]
        self.assertEqual(decision, "BASELINE")
        self.assertEqual(confidence, 0.0)
        self.assertIn("parse failure", reason)
        self.assertFalse(parse_ok)

    def test_last_step_draft_vote_overrides_selector(self):
        class DraftVoteAgent(FakeDraftStepAgent):
            def generate_draft(self, problem):
                del problem
                return "draft prompt", "Draft solution.\nFinal Answer: 371"

            def segment_draft(self, problem, draft):
                del problem, draft
                return (
                    [StepPlan("Verify and output Final Answer: <answer>.", True)],
                    VALID_PLAN,
                    True,
                    "",
                )

            def generate_step_candidate(
                self, problem, accepted, step, *, hint="", source
            ):
                del problem, accepted, step, hint
                answer = "371" if source == "HINTED" else "736"
                return f"{source} prompt", f"Checked.\nFinal Answer: {answer}"

            def select_step(self, *args, **kwargs):
                return DraftStepAgent.select_step(self, *args, **kwargs)

            def _call(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("draft vote should skip the selector")

        trajectory = DraftVoteAgent().run("dummy", gold="371", mode="full")
        self.assertEqual(trajectory.final_answer, "371")
        self.assertEqual(trajectory.steps[0].decision, "HINTED")
        self.assertTrue(trajectory.steps[0].draft_vote)
        self.assertEqual(trajectory.em, 1)
        self.assertEqual(trajectory.harmed, 0)


if __name__ == "__main__":
    unittest.main()
