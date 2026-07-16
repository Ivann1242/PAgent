import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONE = ROOT / "HintFlow_one"
for path in (ROOT, ONE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from one_agent import HintFlowOneAgent, Selection, parse_selection


class FakeOneAgent(HintFlowOneAgent):
    def __init__(self, baseline, challenger, selection=None, **kwargs):
        super().__init__(
            orch_url="http://unused",
            solver_url="http://unused",
            **kwargs,
        )
        self._baseline = baseline
        self._challenger = challenger
        self._selection = selection or Selection("KEEP", 1.0)
        self.hints = ["Check arithmetic carefully."]

    def generate_baseline(self, problem, *, gold=""):
        del problem
        return self._candidate(
            source="BASELINE",
            prompt="p",
            solution=self._baseline,
            gold=gold,
        )

    def generate_hint(self, problem):
        del problem
        hint = self.hints[0]
        return hint, hint, True

    def generate_challenger(self, problem, hint, *, gold=""):
        del problem, hint
        return self._candidate(
            source="FF_CHALLENGER",
            prompt="p",
            solution=self._challenger,
            gold=gold,
        )

    def select(self, problem, incumbent, challenger):
        del problem, incumbent, challenger
        return self._selection


class HintFlowOneTest(unittest.TestCase):
    def test_parse_selection_defaults_to_keep(self):
        self.assertEqual(parse_selection("[]").decision, "KEEP")

    def test_keep_preserves_correct_baseline(self):
        agent = FakeOneAgent(
            "Final Answer: 7",
            "Final Answer: 9",
            Selection("KEEP", 0.99),
            selector_mode="orch",
        )
        traj = agent.run("dummy", gold="7")
        self.assertEqual(traj.baseline_em, 1)
        self.assertEqual(traj.em, 1)
        self.assertEqual(traj.harmed, 0)

    def test_replace_recovers_wrong_baseline(self):
        agent = FakeOneAgent(
            "Final Answer: 9",
            "Final Answer: 7",
            Selection("REPLACE", 0.95),
            selector_mode="orch",
            replace_threshold=0.90,
        )
        traj = agent.run("dummy", gold="7")
        self.assertEqual(traj.baseline_em, 0)
        self.assertEqual(traj.em, 1)
        self.assertEqual(traj.recovered, 1)

    def test_low_confidence_replace_is_ignored(self):
        agent = FakeOneAgent(
            "Final Answer: 7",
            "Final Answer: 9",
            Selection("REPLACE", 0.50),
            replace_threshold=0.90,
        )
        traj = agent.run("dummy", gold="7")
        self.assertEqual(traj.em, 1)
        self.assertEqual(traj.harmed, 0)


if __name__ == "__main__":
    unittest.main()
