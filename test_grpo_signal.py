"""Unit checks for GRPO advantage / loss signal."""

from __future__ import annotations

import torch

from grpo import (
    grpo_loss,
    group_advantages,
    virtual_positive_grpo_advantages,
)


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


def test_virtual_positive_grpo_mixed_is_standard_grpo():
    rewards = [1.0, 1.0, 0.0, 0.0]
    adv, mean_r, std_r, group_type = virtual_positive_grpo_advantages(rewards)
    expected = (torch.tensor(rewards) - mean_r) / std_r
    assert group_type == "mixed"
    assert torch.allclose(adv, expected, atol=1e-5)


def test_virtual_positive_grpo_all_correct_is_kl_only():
    adv, mean_r, std_r, group_type = virtual_positive_grpo_advantages([1.0] * 8)
    assert group_type == "all_correct"
    assert mean_r == 1.0
    assert std_r == 0.0
    assert torch.count_nonzero(adv) == 0


def test_virtual_positive_grpo_all_wrong_is_negative():
    adv, mean_r, std_r, group_type = virtual_positive_grpo_advantages([0.0] * 8)
    assert group_type == "all_wrong"
    assert mean_r == 0.0
    assert std_r == 0.0
    expected = -1.0 / (8.0 ** 0.5)
    assert torch.allclose(adv, torch.full((8,), expected), atol=1e-5)


def test_tied_fractional_group_is_kl_only():
    adv, _, _, group_type = virtual_positive_grpo_advantages([1.0 / 3.0] * 8)
    assert group_type == "tied"
    assert torch.count_nonzero(adv) == 0


def test_zero_advantage_can_still_have_kl_gradient():
    cur_lp = torch.tensor([-1.8, -2.2], requires_grad=True)
    old_lp = cur_lp.detach().clone()
    ref_lp = torch.tensor([-2.0, -2.0])
    loss, stats = grpo_loss(cur_lp, old_lp, ref_lp, 0.0, beta=0.02)
    loss.backward()
    assert stats["kl"] > 0
    assert cur_lp.grad is not None
    assert float(cur_lp.grad.abs().sum()) > 0


if __name__ == "__main__":
    test_rloo_binary_rewards_have_signal()
    test_tied_rewards_no_signal()
    test_pg_loss_nonzero_with_centered_advantages()
    test_virtual_positive_grpo_mixed_is_standard_grpo()
    test_virtual_positive_grpo_all_correct_is_kl_only()
    test_virtual_positive_grpo_all_wrong_is_negative()
    test_tied_fractional_group_is_kl_only()
    test_zero_advantage_can_still_have_kl_gradient()
    print("ok")
