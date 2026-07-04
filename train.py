"""GRPO training for Qwen4B action router (LoRA)."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import torch
from openai import OpenAI
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    CLIP_RANGE,
    GRAD_ACCUM_GROUPS,
    KL_BETA,
    K,
    LR,
    MIN_UNIQUE_ACTIONS,
    ROLLOUT_WORKERS,
    SMALL_TEMP_TRAIN,
    Config,
)
from core import (
    append_jsonl,
    build_small_prompt,
    format_router_input,
    load_jsonl,
    parse_action,
    rollout,
)
from grpo import completion_logprobs, encode_prompt_completion, grpo_loss, group_advantages


def _generate_one(model, tokenizer, prompt, *, max_new_tokens, temperature, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def _generate_group(
    model, tokenizer, prompt, *,
    k, max_new_tokens, temperature, device,
    min_unique: int = 1,
    max_rounds: int = 3,
):
    was_train = model.training
    model.eval()
    try:
        completions: list[str] = []
        for round_i in range(max_rounds):
            temp = temperature * (1.0 + 0.35 * round_i)
            while len(completions) < k:
                completions.append(_generate_one(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens, temperature=temp, device=device,
                ))
            actions = {parse_action(c)[0] for c in completions}
            if len(actions) >= min_unique or round_i + 1 >= max_rounds:
                return completions[:k]
            completions = []
    finally:
        if was_train:
            model.train()
    return completions[:k]


def _logprobs(model, tokenizer, prompt, completion, device, *, grad=False, no_adapter=False):
    input_ids, start = encode_prompt_completion(tokenizer, prompt, completion, device)
    ctx = torch.enable_grad() if grad else torch.inference_mode()
    adapter_ctx = model.disable_adapter() if no_adapter else nullcontext()
    with ctx, adapter_ctx:
        return completion_logprobs(model, input_ids, start)


def _maybe_empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _rollout_one(answer_client, answer_model, problem, gold, comp):
    action, parse_ok = parse_action(comp)
    r = rollout(
        answer_client, answer_model, problem, gold, action,
        small_output=comp, parse_ok=parse_ok,
    )
    return r


def _rollout_group(answer_client, answer_model, problem, gold, completions, *, workers):
    if workers <= 1 or len(completions) <= 1:
        return [
            _rollout_one(answer_client, answer_model, problem, gold, comp)
            for comp in completions
        ]
    with ThreadPoolExecutor(max_workers=min(workers, len(completions))) as pool:
        return list(pool.map(
            lambda comp: _rollout_one(answer_client, answer_model, problem, gold, comp),
            completions,
        ))


def _optimizer_step(model, optimizer) -> float:
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def train(
    cfg: Config,
    *,
    limit=None,
    max_steps=None,
    k=K,
    lr=LR,
    gpu="1",
    grad_accum_groups=GRAD_ACCUM_GROUPS,
    min_unique_actions=MIN_UNIQUE_ACTIONS,
    rollout_workers=ROLLOUT_WORKERS,
) -> Path:
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)

    rows = load_jsonl(cfg.train_file)
    if limit:
        rows = rows[:limit]

    answer_client = OpenAI(base_url=cfg.answer_url, api_key="EMPTY")
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg.rollout_log.exists():
        cfg.rollout_log.unlink()

    tokenizer = AutoTokenizer.from_pretrained(cfg.router_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.router_base,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    step = 0
    skipped = 0
    pending_groups = 0
    grad_scale = 1.0 / max(grad_accum_groups * k, 1)

    def flush_optimizer(*, force: bool = False) -> float | None:
        nonlocal pending_groups
        if pending_groups == 0:
            return None
        if not force and pending_groups < grad_accum_groups:
            return None
        grad_norm = _optimizer_step(model, optimizer)
        pending_groups = 0
        _maybe_empty_cache(device)
        return grad_norm

    for row in rows:
        problem, gold = row["problem"], row["gold"]
        small_prompt = format_router_input(tokenizer, build_small_prompt(problem))
        completions = _generate_group(
            model, tokenizer, small_prompt,
            k=k, max_new_tokens=128, temperature=SMALL_TEMP_TRAIN, device=device,
            min_unique=min_unique_actions,
        )
        _maybe_empty_cache(device)

        old_lps, ref_lps, rewards, ems = [], [], [], []
        for comp in completions:
            old_lps.append(_logprobs(model, tokenizer, small_prompt, comp, device).detach())
            ref_lps.append(_logprobs(model, tokenizer, small_prompt, comp, device, no_adapter=True).detach())
        _maybe_empty_cache(device)

        rollout_rows = _rollout_group(
            answer_client, cfg.answer_model, problem, gold, completions,
            workers=rollout_workers,
        )
        for r in rollout_rows:
            append_jsonl(cfg.rollout_log, r)
            rewards.append(r["reward"])
            ems.append(r["em"])

        advantages, mean_r, std_r, has_signal = group_advantages(rewards)
        invalid = sum(parse_action(c)[1] is False for c in completions) / len(completions)
        unique_actions = len({parse_action(c)[0] for c in completions})
        actions = [parse_action(c)[0] for c in completions]

        step += 1
        if not has_signal:
            skipped += 1
            print(
                f"step={step} reward={mean_r:.3f}±{std_r:.3f} em={sum(ems)/len(ems):.0%} "
                f"invalid={invalid:.0%} unique={unique_actions} SKIP actions={actions}",
                flush=True,
            )
        else:
            pg_terms = []
            for comp, adv, old_lp, ref_lp in zip(completions, advantages.tolist(), old_lps, ref_lps):
                cur_lp = _logprobs(model, tokenizer, small_prompt, comp, device, grad=True)
                loss, stats = grpo_loss(cur_lp, old_lp, ref_lp, adv, clip=CLIP_RANGE, beta=KL_BETA)
                (loss * grad_scale).backward()
                pg_terms.append(stats["pg"])
                del cur_lp, loss

            pending_groups += 1
            grad_norm = flush_optimizer(force=False)
            _maybe_empty_cache(device)
            pg_mean = sum(pg_terms) / len(pg_terms)
            grad_msg = f" grad={grad_norm:.4f}" if grad_norm is not None else ""
            print(
                f"step={step} reward={mean_r:.3f}±{std_r:.3f} em={sum(ems)/len(ems):.0%} "
                f"invalid={invalid:.0%} unique={unique_actions} pg={pg_mean:.3f}{grad_msg} "
                f"actions={actions}",
                flush=True,
            )

        if max_steps and step >= max_steps:
            break

    final_grad = flush_optimizer(force=True)
    if final_grad is not None:
        print(f"final flush grad={final_grad:.4f}", flush=True)

    print(f"skipped_groups={skipped}/{step} ({100*skipped/max(step,1):.1f}%)", flush=True)

    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.adapter_dir)
    tokenizer.save_pretrained(cfg.adapter_dir)
    print(f"adapter -> {cfg.adapter_dir}")
    return cfg.adapter_dir


def merge(cfg: Config) -> Path:
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(cfg.router_base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.router_base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, cfg.adapter_dir).merge_and_unload()
    cfg.merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.merged_dir)
    tok.save_pretrained(cfg.merged_dir)
    print(f"merged -> {cfg.merged_dir}")
    return cfg.merged_dir
