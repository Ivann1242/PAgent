"""GRPO loss and group advantage computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def group_advantages(
    rewards: list[float],
    *,
    eps: float = 1e-6,
    method: str = "rloo",
) -> tuple[torch.Tensor, float, float, bool]:
    """Return (advantages, mean_reward, std_reward, has_signal).

    Uses leave-one-out (RLOO) baselines by default so binary EM rewards still
    produce non-zero advantages whenever not all rollouts tie.
    """
    t = torch.tensor(rewards, dtype=torch.float32)
    mean_r = float(t.mean())
    std_r = float(t.std(unbiased=False))
    n = t.numel()

    if n <= 1:
        return torch.zeros_like(t), mean_r, std_r, False

    if method == "grpo":
        adv = (t - mean_r) / (std_r + eps)
    else:
        # RLOO: A_i = r_i - mean(r_{-i})
        total = t.sum()
        adv = t - (total - t) / (n - 1)

    has_signal = std_r > eps and float(adv.abs().max()) > eps
    if not has_signal:
        adv = torch.zeros_like(adv)
    return adv, mean_r, std_r, has_signal


def completion_logprobs(model, input_ids: torch.Tensor, completion_start: int) -> torch.Tensor:
    logits = model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    comp_ids = input_ids[0, completion_start:]
    token_lps = []
    for i, tid in enumerate(comp_ids):
        pos = completion_start - 1 + i
        token_lps.append(log_probs[0, pos, tid])
    if not token_lps:
        return torch.zeros(0, device=input_ids.device)
    return torch.stack(token_lps)


def encode_prompt_completion(tokenizer, prompt: str, completion: str, device: torch.device):
    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    c_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([p_ids, c_ids], dim=1)
    return input_ids, p_ids.shape[1]


def grpo_loss(
    current_lp: torch.Tensor,
    old_lp: torch.Tensor,
    ref_lp: torch.Tensor,
    advantage: float,
    *,
    clip: float = 0.2,
    beta: float = 0.02,
) -> tuple[torch.Tensor, dict]:
    if current_lp.numel() == 0:
        z = current_lp.sum()
        return z, {"clip_ratio": 0.0, "kl": 0.0, "pg": 0.0}

    ratio = torch.exp(current_lp - old_lp)
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    adv = torch.tensor(advantage, device=current_lp.device, dtype=current_lp.dtype)
    pg = -torch.min(ratio * adv, clipped * adv)

    kl = 0.0
    if beta:
        per_kl = torch.exp(ref_lp - current_lp) - (ref_lp - current_lp) - 1.0
        pg = pg + beta * per_kl
        kl = float(per_kl.mean().detach().cpu())

    clipped_frac = float(
        ((ratio < 1.0 - clip) & (adv < 0) | (ratio > 1.0 + clip) & (adv > 0))
        .float().mean().detach().cpu()
    )
    # Sum token losses so centered advantages still yield a non-zero objective.
    pg_scalar = float(pg.sum().detach().cpu())
    return pg.sum(), {"clip_ratio": clipped_frac, "kl": kl, "pg": pg_scalar}
