"""GRPO loss and group advantage computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def virtual_positive_grpo_advantages(
    rewards: list[float],
    *,
    eps: float = 1e-6,
    all_wrong_mode: str = "vp",
) -> tuple[torch.Tensor, float, float, str]:
    """Standard GRPO advantages with explicit handling for degenerate groups.

    Mixed groups use the usual within-group mean/std normalization.
    All-correct groups receive zero policy-gradient advantage (KL-only).
    All-wrong groups append one virtual reward=1 for normalization only; the
    virtual item never participates in backprop.  For K real zero rewards this
    gives every real completion advantage -1/sqrt(K).
    """
    if all_wrong_mode not in {"vp", "off"}:
        raise ValueError("all_wrong_mode must be vp or off")
    t = torch.tensor(rewards, dtype=torch.float32)
    if t.numel() == 0:
        return t, 0.0, 0.0, "empty"

    mean_r = float(t.mean())
    std_r = float(t.std(unbiased=False))
    if bool(torch.all(t >= 1.0 - eps)):
        return torch.zeros_like(t), mean_r, std_r, "all_correct"
    if bool(torch.all(t <= eps)):
        if all_wrong_mode == "off":
            return torch.zeros_like(t), mean_r, std_r, "all_wrong"
        augmented = torch.cat([t, torch.ones(1, dtype=t.dtype)])
        aug_mean = augmented.mean()
        aug_std = augmented.std(unbiased=False)
        adv = (t - aug_mean) / (aug_std + eps)
        return adv, mean_r, std_r, "all_wrong"
    if std_r <= eps:
        # Repeat-averaged rewards can tie at 1/3 or 2/3. There is no relative
        # correctness signal, so keep this group KL-only.
        return torch.zeros_like(t), mean_r, std_r, "tied"

    adv = (t - t.mean()) / (t.std(unbiased=False) + eps)
    return adv, mean_r, std_r, "mixed"


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
    # Normalize within each completion so long hints do not dominate updates.
    pg_scalar = float(pg.mean().detach().cpu())
    return pg.mean(), {"clip_ratio": clipped_frac, "kl": kl, "pg": pg_scalar}
