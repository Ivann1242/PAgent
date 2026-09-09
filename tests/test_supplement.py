"""Run on the server: python -m unittest discover -s tests -p test_supplement.py.

These tests use synthetic data and mocks, never a live model service.
"""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.supplement.gpu import bind, verify_worker
from experiments.supplement.io import write_json
from experiments.supplement.trace import Ledger, BudgetExceeded
from experiments.supplement.analyze import paired_ci, usage
from experiments.supplement.runner import latest_checkpoint


class InfrastructureTests(unittest.TestCase):
    def test_forbidden_gpu(self):
        for physical in (0, 3, -1):
            with self.assertRaises(ValueError):
                bind(physical, {1: "GPU-one", 2: "GPU-two"})
        self.assertEqual(bind(1, {1: "GPU-one", 2: "GPU-two"})["CUDA_VISIBLE_DEVICES"], "GPU-one")

    def test_gpu_binding_cannot_be_overridden(self):
        with patch("experiments.supplement.gpu.inventory", return_value={1: "GPU-one", 2: "GPU-two"}):
            with patch.dict(os.environ, {"PAGENT_PHYSICAL_GPU": "1", "PAGENT_GPU_UUID": "GPU-one", "CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaises(RuntimeError):
                    verify_worker()

    def test_budget_reserves_before_request_and_does_not_refund_error(self):
        budget = Ledger(100)
        budget.reserve(40)
        budget.reserve(60)
        with self.assertRaises(BudgetExceeded):
            budget.reserve(1)
        self.assertEqual(budget.reserved, 100)

    def test_partial_checkpoint_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "snapshots" / "step-0050"
            partial.mkdir(parents=True)
            write_json(partial / "grpo_train_state.json", {"step": 50})
            self.assertIsNone(latest_checkpoint(root))
            complete = root / "snapshots" / "step-0025"
            complete.mkdir()
            write_json(complete / "grpo_train_state.json", {"step": 25})
            write_json(complete / "adapter_config.json", {})
            (complete / "optimizer_state.pt").write_bytes(b"test-only")
            (complete / "COMPLETE").write_text("25")
            self.assertEqual(latest_checkpoint(root), complete)

    def test_pairing_retains_zero_difference(self):
        self.assertEqual(paired_ci([0, 0, 0], repeats=100), [0, 0])

    def test_missing_usage_not_zero_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            path.write_text(json.dumps({"event": "request", "call_id": "x", "request": {"max_tokens": 100}}) + "\n")
            cost = usage([path])
            self.assertFalse(cost["tokens_complete"])
            self.assertEqual(cost["unresolved_requests"], 1)

    def test_exact_dedup_preserves_math_punctuation(self):
        from experiments.supplement.prepare import normalize
        self.assertEqual(normalize("x + 1"), normalize("x+1"))
        self.assertNotEqual(normalize("x+1"), normalize("x-1"))

    def test_interrupted_trial_is_not_automatically_retried(self):
        from experiments.supplement.evaluate import cache
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.json"
            with self.assertRaises(ZeroDivisionError):
                cache(path, lambda: 1 / 0)
            with self.assertRaises(RuntimeError):
                cache(path, lambda: {"em": 1})
            self.assertFalse(path.exists())


class AdvantageTests(unittest.TestCase):
    def test_vp_and_off_differ_only_for_all_wrong(self):
        import torch
        from grpo import virtual_positive_grpo_advantages as advantages
        for rewards in ([0, 1], [1, 1], [.5, .5], [], [.2, .8]):
            on = advantages(rewards, all_wrong_mode="vp")
            off = advantages(rewards, all_wrong_mode="off")
            self.assertTrue(torch.equal(on[0], off[0]))
            self.assertEqual(on[3], off[3])
        for k in (1, 2, 8):
            on = advantages([0] * k)[0]
            self.assertEqual(len(on), k)  # no virtual completion in loss
            self.assertTrue(torch.allclose(on, torch.full((k,), -1 / (math.sqrt(k) + (k + 1) * 1e-6))))
            self.assertTrue(torch.equal(advantages([0] * k, all_wrong_mode="off")[0], torch.zeros(k)))

    def test_off_all_wrong_keeps_kl_gradient(self):
        import torch
        from grpo import grpo_loss
        current = torch.tensor([-.5, -.6], requires_grad=True)
        old = torch.tensor([-.7, -.8])
        ref = torch.tensor([-.9, -1.])
        loss, _ = grpo_loss(current, old, ref, 0.0, beta=.02)
        loss.backward()
        self.assertGreater(current.grad.abs().sum().item(), 0)

    def test_equal_constant_penalty_has_identical_gradient(self):
        import torch
        from grpo import grpo_loss, virtual_positive_grpo_advantages
        k = 8
        first = torch.tensor([-.5, -.6], requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        old, ref = torch.tensor([-.7, -.8]), torch.tensor([-.9, -1.])
        vp = virtual_positive_grpo_advantages([0] * k)[0][0].item()
        constant = -1 / (math.sqrt(k) + (k + 1) * 1e-6)
        grpo_loss(first, old, ref, vp)[0].backward()
        grpo_loss(second, old, ref, constant)[0].backward()
        self.assertTrue(torch.allclose(first.grad, second.grad))


if __name__ == "__main__":
    unittest.main()
