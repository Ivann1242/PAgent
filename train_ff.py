"""GRPO training for free-form prompt optimizer (Qwen4B LoRA)."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import torch
from openai import OpenAI
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ANSWER_URLS, CLIP_RANGE, KL_BETA, LR, SMALL_TEMP_TRAIN, Config
from core import (
    append_jsonl,
    build_optimizer_prompt,
    format_router_input,
    load_jsonl,
    parse_optimizer_output,
    rollout_ff,
)
from grpo import completion_logprobs, encode_prompt_completion, grpo_loss, group_advantages
from label import _AnswererPool


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
            hints = {parse_optimizer_output(c)[0] for c in completions}
            if len(hints) >= min_unique or round_i + 1 >= max_rounds:
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


def train_ff(
    cfg: Config,
    *,
    batch_size: int = 64,
    max_steps: int = 10,
    k: int = 8,
    lr: float = LR,
    gpu: str = "1",
    rollout_workers: int = 32,
    min_unique: int = 1,
    seed: int = 42,
    answer_urls: list[str] | None = None,
) -> Path:
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    rows = load_jsonl(cfg.train_file)
    if len(rows) < batch_size:
        raise SystemExit(f"train_file has {len(rows)} rows, need batch_size={batch_size}")

    pool = _AnswererPool(answer_urls or cfg.answer_urls, cfg.answer_model)
    cfg.ff_adapter_dir.mkdir(parents=True, exist_ok=True)
    if cfg.ff_rollout_log.exists():
        cfg.ff_rollout_log.unlink()

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

    grad_scale = 1.0 / max(batch_size * k, 1)
    skipped_groups = 0
    total_groups = 0
    cursor = 0

    def _rollout_task(item):
        row, comp = item
        hint, _ = parse_optimizer_output(comp)
        client = pool.next_client()
        r = rollout_ff(
            client, cfg.answer_model, row["problem"], row["gold"], hint,
            small_output=comp,
        )
        r["id"] = row["id"]
        return row["id"], comp, r

    for step in range(1, max_steps + 1):
        batch = [rows[(cursor + i) % len(rows)] for i in range(batch_size)]
        cursor = (cursor + batch_size) % len(rows)

        groups: list[dict] = []
        for row in batch:
            opt_prompt = format_router_input(tokenizer, build_optimizer_prompt(row["problem"]))
            completions = _generate_group(
                model, tokenizer, opt_prompt,
                k=k, max_new_tokens=256, temperature=SMALL_TEMP_TRAIN, device=device,
                min_unique=min_unique,
            )
            old_lps, ref_lps = [], []
            for comp in completions:
                old_lps.append(_logprobs(model, tokenizer, opt_prompt, comp, device).detach())
                ref_lps.append(
                    _logprobs(model, tokenizer, opt_prompt, comp, device, no_adapter=True).detach()
                )
            groups.append({
                "row": row,
                "opt_prompt": opt_prompt,
                "completions": completions,
                "old_lps": old_lps,
                "ref_lps": ref_lps,
            })
        _maybe_empty_cache(device)

        tasks = [
            (g["row"], comp)
            for g in groups for comp in g["completions"]
        ]
        rollout_map: dict[tuple[int, str], dict] = {}
        workers = min(rollout_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for qid, comp, r in ex.map(_rollout_task, tasks):
                rollout_map[(qid, comp)] = r
                append_jsonl(cfg.ff_rollout_log, r)

        step_skipped = 0
        step_em = 0.0
        step_pg = 0.0
        n_pg = 0
        for g in groups:
            row = g["row"]
            rewards, ems = [], []
            for comp in g["completions"]:
                r = rollout_map[(row["id"], comp)]
                rewards.append(r["reward"])
                ems.append(r["em"])

            advantages, mean_r, std_r, has_signal = group_advantages(rewards)
            total_groups += 1
            step_em += sum(ems) / len(ems)
            invalid = sum(parse_optimizer_output(c)[1] is False for c in g["completions"]) / len(g["completions"])

            if not has_signal:
                step_skipped += 1
                skipped_groups += 1
                continue

            for comp, adv, old_lp, ref_lp in zip(
                g["completions"], advantages.tolist(), g["old_lps"], g["ref_lps"],
            ):
                cur_lp = _logprobs(model, tokenizer, g["opt_prompt"], comp, device, grad=True)
                loss, stats = grpo_loss(cur_lp, old_lp, ref_lp, adv, clip=CLIP_RANGE, beta=KL_BETA)
                (loss * grad_scale).backward()
                step_pg += stats["pg"]
                n_pg += 1
                del cur_lp, loss

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _maybe_empty_cache(device)

        print(
            f"step={step}/{max_steps} batch={batch_size} k={k} "
            f"em={step_em/batch_size:.0%} skip_groups={step_skipped}/{batch_size} "
            f"pg={step_pg/max(n_pg,1):.3f} grad={grad_norm:.4f}",
            flush=True,
        )

    print(
        f"skipped_groups={skipped_groups}/{total_groups} "
        f"({100*skipped_groups/max(total_groups,1):.1f}%)",
        flush=True,
    )
    model.save_pretrained(cfg.ff_adapter_dir)
    tokenizer.save_pretrained(cfg.ff_adapter_dir)
    print(f"adapter -> {cfg.ff_adapter_dir}")
    return cfg.ff_adapter_dir


def merge_ff(cfg: Config) -> Path:
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(cfg.router_base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.router_base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, cfg.ff_adapter_dir).merge_and_unload()
    cfg.ff_merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.ff_merged_dir)
    tok.save_pretrained(cfg.ff_merged_dir)
    print(f"merged -> {cfg.ff_merged_dir}")
    return cfg.ff_merged_dir
