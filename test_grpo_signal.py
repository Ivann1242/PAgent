"""Unit checks for GRPO advantage / loss signal."""

from __future__ import annotations

import torch

from grpo import grpo_loss, group_advantages


def test_rloo_binary_rewards_have_signal():
    rewards = [1.0, 1.0, 0.0, 0.0]
    adv, mean_r, std_r, has_signal = group_advantages(rewards)
    assert has_signal
    assert std_r > 0
    assert mean_r == 0.5
    assert adv[0] > 0 and adv[2] < 0


def test_tied_rewards_no_signal():
    rewards = [1.0, 1.0, 1.0, 1.0]
    adv, _, std_r, has_signal = group_advantages(rewards)
    assert std_r == 0.0
    assert not has_signal
    assert float(adv.abs().max()) == 0.0


def test_pg_loss_nonzero_with_centered_advantages():
    rewards = [1.0, 1.0, 0.0, 0.0]
    adv, _, _, has_signal = group_advantages(rewards)
    assert has_signal
    old_lps = [torch.tensor([-2.0]), torch.tensor([-2.5]), torch.tensor([-3.0]), torch.tensor([-3.5])]
    total = torch.tensor(0.0)
    for a, old_lp in zip(adv.tolist(), old_lps):
        cur_lp = old_lp.clone().requires_grad_(True)
        ref_lp = old_lp.clone()
        loss, _ = grpo_loss(cur_lp, old_lp, ref_lp, a)
        total = total + loss
        loss.backward()
        assert cur_lp.grad is not None
        assert float(cur_lp.grad.abs().sum()) > 0.0
    assert float(total.detach()) != 0.0


if __name__ == "__main__":
    test_rloo_binary_rewards_have_signal()
    test_tied_rewards_no_signal()
    test_pg_loss_nonzero_with_centered_advantages()
    print("ok")
