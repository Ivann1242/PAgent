"""GRPO training for free-form prompt optimizer (Qwen4B LoRA)."""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
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
from grpo import (
    completion_logprobs,
    encode_prompt_completion,
    grpo_loss,
    virtual_positive_grpo_advantages,
)
from label import _AnswererPool


def _generate_k_samples(
    model, tokenizer, prompt, *, k, max_new_tokens, temperature, device,
) -> list[str]:
    """Sample K independent completions for one prompt in a single generate call."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True)
        for i in range(out.shape[0])
    ]


def _generate_batch_groups(
    model, tokenizer, prompts: list[str], *,
    k, max_new_tokens, temperature, device,
) -> list[list[str]]:
    """Sample K completions per prompt; batch prompts when len > 1."""
    if not prompts:
        return []
    if len(prompts) == 1:
        return [_generate_k_samples(
            model, tokenizer, prompts[0],
            k=k, max_new_tokens=max_new_tokens, temperature=temperature, device=device,
        )]

    _maybe_empty_cache(device)
    pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(device)
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    num_return_sequences=k,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            _maybe_empty_cache(device)
            if len(prompts) <= 1:
                raise
            mid = max(1, len(prompts) // 2)
            return (
                _generate_batch_groups(
                    model, tokenizer, prompts[:mid],
                    k=k, max_new_tokens=max_new_tokens, temperature=temperature, device=device,
                )
                + _generate_batch_groups(
                    model, tokenizer, prompts[mid:],
                    k=k, max_new_tokens=max_new_tokens, temperature=temperature, device=device,
                )
            )
        prompt_len = inputs["input_ids"].shape[1]
        groups: list[list[str]] = []
        for b in range(len(prompts)):
            groups.append([
                tokenizer.decode(out[b * k + j, prompt_len:], skip_special_tokens=True)
                for j in range(k)
            ])
        return groups
    finally:
        tokenizer.padding_side = pad_side


def _generate_group(
    model, tokenizer, prompt, *,
    k, max_new_tokens, temperature, device,
    min_unique: int = 1,
    max_rounds: int = 3,
):
    was_train = model.training
    model.eval()
    try:
        for round_i in range(max_rounds):
            temp = temperature * (1.0 + 0.35 * round_i)
            completions = _generate_k_samples(
                model, tokenizer, prompt,
                k=k, max_new_tokens=max_new_tokens, temperature=temp, device=device,
            )
            hints = {parse_optimizer_output(c)[0] for c in completions}
            if len(hints) >= min_unique or round_i + 1 >= max_rounds:
                return completions[:k]
    finally:
        if was_train:
            model.train()
    return completions[:k]


def _logprobs(model, tokenizer, prompt, completion, device, *, grad=False):
    input_ids, start = encode_prompt_completion(tokenizer, prompt, completion, device)
    ctx = torch.enable_grad() if grad else torch.inference_mode()
    with ctx:
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
    gen_batch_size: int = 4,
    min_unique: int = 1,
    seed: int = 42,
    answer_urls: list[str] | None = None,
    data_file: Path | None = None,
    init_adapter_dir: Path | None = None,
    adapter_dir: Path | None = None,
    rollout_log: Path | None = None,
    start_step: int = 1,
    reward_repeats: int = 1,
    reward_max_tokens: int = 8192,
    reward_temperature: float = 0.0,
    reference_adapter_dir: Path | None = None,
    checkpoint_every: int = 10,
    hf_repo: str | None = None,
) -> Path:
    import json
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    rows = load_jsonl(data_file or cfg.train_file)
    if len(rows) < batch_size:
        raise SystemExit(f"train data has {len(rows)} rows, need batch_size={batch_size}")

    pool = _AnswererPool(answer_urls or cfg.answer_urls, cfg.answer_model)
    adapter_dir = Path(adapter_dir or cfg.ff_adapter_dir)
    rollout_log = Path(rollout_log or cfg.ff_rollout_log)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    state_path = adapter_dir / "grpo_train_state.json"
    if start_step > 1 and state_path.exists():
        state = json.loads(state_path.read_text())
        cursor = int(state.get("cursor", 0))
        print(f"resume from step={start_step} cursor={cursor}", flush=True)
    else:
        cursor = 0
        start_step = 1
        if rollout_log.exists():
            rollout_log.unlink()

    tokenizer = AutoTokenizer.from_pretrained(cfg.router_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.router_base,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    if init_adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, Path(init_adapter_dir), is_trainable=True)
        print(f"init adapter -> {init_adapter_dir}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    if reference_adapter_dir is None:
        reference_adapter_dir = init_adapter_dir
    if reference_adapter_dir is None:
        raise SystemExit(
            "virtual-positive GRPO requires --reference-adapter "
            "(normally the SFT initialization)"
        )
    from peft import PeftModel
    ref_base = AutoModelForCausalLM.from_pretrained(
        cfg.router_base,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    ref_model = PeftModel.from_pretrained(
        ref_base, Path(reference_adapter_dir), is_trainable=False,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    print(f"fixed reference adapter -> {reference_adapter_dir}", flush=True)
    reward_repeats = max(1, int(reward_repeats))
    print(
        f"hint gen: batched num_return_sequences={k}, gen_batch_size={gen_batch_size}",
        flush=True,
    )
    print(
        f"reward: repeats={reward_repeats} max_tokens={reward_max_tokens} "
        f"temperature={reward_temperature} urls={len(pool.urls)} "
        f"rollout_workers={rollout_workers}",
        flush=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr,
    )
    optimizer.zero_grad(set_to_none=True)
    optimizer_state_path = adapter_dir / "optimizer_state.pt"
    if start_step > 1 and optimizer_state_path.exists():
        saved = torch.load(optimizer_state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        if "python_rng" in saved:
            random.setstate(saved["python_rng"])
        if "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device=device)
        print(f"optimizer/RNG resumed from {optimizer_state_path}", flush=True)

    grad_scale = 1.0 / max(batch_size * k, 1)
    skipped_groups = 0
    total_groups = 0
    hf_repo_initialized = False

    def _save_step(step: int) -> None:
        nonlocal hf_repo_initialized
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        state = {
            "step": step,
            "cursor": cursor,
            "batch_size": batch_size,
            "k": k,
            "gen_batch_size": gen_batch_size,
            "reward_repeats": reward_repeats,
            "reward_max_tokens": reward_max_tokens,
            "reward_temperature": reward_temperature,
            "advantage_method": "virtual_positive_grpo",
            "reference_adapter": str(reference_adapter_dir),
            "kl_beta": KL_BETA,
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        torch.save({
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, optimizer_state_path)
        if hf_repo and checkpoint_every > 0 and step % checkpoint_every == 0:
            from huggingface_hub import HfApi, create_repo

            token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
            if not token:
                env_file = Path("/home/ivaning/prompt-r1r/Prompt-R1/.env")
                if env_file.exists():
                    values = {}
                    for line in env_file.read_text().splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        values[key.strip()] = value.strip().strip('"').strip("'")
                    token = values.get("HF_WRITE_TOKEN") or values.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF checkpoint requested but no HF token found")
            if not hf_repo_initialized:
                create_repo(
                    hf_repo, repo_type="model", exist_ok=True,
                    private=False, token=token,
                )
                hf_repo_initialized = True
            api = HfApi(token=token)
            api.upload_folder(
                folder_path=str(adapter_dir),
                repo_id=hf_repo,
                repo_type="model",
                path_in_repo=f"step-{step:04d}",
                ignore_patterns=["README.md"],
                commit_message=f"GRPO checkpoint step {step}",
                token=token,
            )
            print(
                f"HF checkpoint step={step} -> "
                f"https://huggingface.co/{hf_repo}/tree/main/step-{step:04d}",
                flush=True,
            )

    def _rollout_once(row: dict, hint: str, *, small_output: str = "") -> dict:
        client = pool.next_client()
        r = rollout_ff(
            client, cfg.answer_model, row["problem"], row["gold"], hint,
            small_output=small_output,
            max_tokens=reward_max_tokens,
            temperature=reward_temperature,
        )
        r["id"] = row["id"]
        r["reward_temperature"] = reward_temperature
        r["reward_max_tokens"] = reward_max_tokens
        r["reward_repeats"] = reward_repeats
        return r

    for step in range(start_step, max_steps + 1):
        step_t0 = time.monotonic()
        batch = [rows[(cursor + i) % len(rows)] for i in range(batch_size)]
        cursor = (cursor + batch_size) % len(rows)

        groups: list[dict] = []
        gen_t0 = time.monotonic()
        was_train = model.training
        model.eval()
        try:
            for chunk_start in range(0, len(batch), gen_batch_size):
                chunk = batch[chunk_start: chunk_start + gen_batch_size]
                prompts = [
                    format_router_input(tokenizer, build_optimizer_prompt(row["problem"]))
                    for row in chunk
                ]
                batch_completions = _generate_batch_groups(
                    model, tokenizer, prompts,
                    k=k, max_new_tokens=256, temperature=SMALL_TEMP_TRAIN, device=device,
                )
                for row, opt_prompt, completions in zip(chunk, prompts, batch_completions):
                    hints = {parse_optimizer_output(c)[0] for c in completions}
                    if min_unique > 1 and len(hints) < min_unique:
                        completions = _generate_group(
                            model, tokenizer, opt_prompt,
                            k=k, max_new_tokens=256, temperature=SMALL_TEMP_TRAIN, device=device,
                            min_unique=min_unique,
                        )
                    old_lps, ref_lps = [], []
                    for comp in completions:
                        old_lps.append(_logprobs(model, tokenizer, opt_prompt, comp, device).detach())
                        ref_lps.append(
                            _logprobs(
                                ref_model, tokenizer, opt_prompt, comp, device,
                            ).detach()
                        )
                    groups.append({
                        "row": row,
                        "opt_prompt": opt_prompt,
                        "completions": completions,
                        "old_lps": old_lps,
                        "ref_lps": ref_lps,
                    })
        finally:
            if was_train:
                model.train()
        _maybe_empty_cache(device)
        gen_sec = time.monotonic() - gen_t0

        # Score unique (qid, hint) only once × reward_repeats, then broadcast.
        # Identical / colliding hints across K samples share the same averaged EM.
        unique_jobs: dict[tuple[int, str], dict] = {}
        comp_to_hint: dict[tuple[int, str], str] = {}
        for g in groups:
            row = g["row"]
            for comp in g["completions"]:
                hint, _ = parse_optimizer_output(comp)
                hint_key = hint.strip()
                comp_to_hint[(row["id"], comp)] = hint_key
                ukey = (row["id"], hint_key)
                if ukey not in unique_jobs:
                    unique_jobs[ukey] = {"row": row, "hint": hint_key, "small_output": comp}

        tasks = [
            (ukey, job, rep_i)
            for ukey, job in unique_jobs.items()
            for rep_i in range(reward_repeats)
        ]
        em_lists: dict[tuple[int, str], list[int]] = {ukey: [] for ukey in unique_jobs}
        reward_lists: dict[tuple[int, str], list[float]] = {
            ukey: [] for ukey in unique_jobs
        }
        error_counts: dict[tuple[int, str], int] = {
            ukey: 0 for ukey in unique_jobs
        }
        workers = max(1, min(rollout_workers, len(tasks)))
        n_oss_calls = len(tasks)
        n_unique_hints = len(unique_jobs)

        def _rollout_task(item):
            ukey, job, rep_i = item
            try:
                r = _rollout_once(
                    job["row"], job["hint"], small_output=job["small_output"],
                )
                compact = {
                    "id": r["id"],
                    "repeat_i": rep_i,
                    "hint": job["hint"],
                    "em": int(r["em"]),
                    "reward": float(r["reward"]),
                    "format_ok": int(r.get("format_ok") or 0),
                    "pred_answer": r.get("pred_answer", ""),
                    "reward_temperature": reward_temperature,
                    "reward_max_tokens": reward_max_tokens,
                    "error": None,
                }
                return ukey, compact
            except Exception as exc:  # transient rollout failures are masked
                return ukey, {
                    "id": job["row"]["id"],
                    "repeat_i": rep_i,
                    "hint": job["hint"],
                    "em": None,
                    "reward": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        reward_t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ukey, r in ex.map(_rollout_task, tasks):
                if r["error"] is None:
                    em_lists[ukey].append(int(r["em"]))
                    # Uses anti-leak-aware rollout reward, not raw EM.
                    reward_lists[ukey].append(float(r["reward"]))
                else:
                    error_counts[ukey] += 1
                append_jsonl(rollout_log, r)
        reward_sec = time.monotonic() - reward_t0

        reward_map: dict[tuple[int, str], float | None] = {}
        em_map: dict[tuple[int, str], float | None] = {}
        repeat_disagreements = 0
        for ukey, ems in em_lists.items():
            rewards_for_hint = reward_lists[ukey]
            if not ems or not rewards_for_hint:
                reward_map[ukey] = None
                em_map[ukey] = None
                continue
            if len(set(ems)) > 1:
                repeat_disagreements += 1
            reward_map[ukey] = float(
                sum(rewards_for_hint) / len(rewards_for_hint)
            )
            em_map[ukey] = float(sum(ems) / len(ems))

        update_t0 = time.monotonic()
        step_skipped = 0
        step_em = 0.0
        step_pg = 0.0
        step_kl = 0.0
        step_clip = 0.0
        n_pg = 0
        group_types = {
            "all_correct": 0, "all_wrong": 0, "mixed": 0,
            "tied": 0, "empty": 0,
        }
        for g in groups:
            row = g["row"]
            valid = []
            for comp, old_lp, ref_lp in zip(
                g["completions"], g["old_lps"], g["ref_lps"],
            ):
                hint_key = comp_to_hint[(row["id"], comp)]
                ukey = (row["id"], hint_key)
                reward = reward_map[ukey]
                mean_em = em_map[ukey]
                if reward is not None and mean_em is not None:
                    valid.append((comp, old_lp, ref_lp, reward, mean_em))

            rewards = [item[3] for item in valid]
            ems = [item[4] for item in valid]
            advantages, mean_r, std_r, group_type = (
                virtual_positive_grpo_advantages(rewards)
            )
            group_types[group_type] = group_types.get(group_type, 0) + 1
            total_groups += 1
            step_em += sum(ems) / len(ems) if ems else 0.0

            if group_type == "empty":
                step_skipped += 1
                skipped_groups += 1
                continue

            # all-correct and fractional tied groups have advantage=0 and are
            # intentionally retained as fixed-SFT KL-only examples.
            for (comp, old_lp, ref_lp, _, _), adv in zip(
                valid, advantages.tolist(),
            ):
                cur_lp = _logprobs(model, tokenizer, g["opt_prompt"], comp, device, grad=True)
                loss, stats = grpo_loss(cur_lp, old_lp, ref_lp, adv, clip=CLIP_RANGE, beta=KL_BETA)
                (loss * grad_scale).backward()
                step_pg += stats["pg"]
                step_kl += stats["kl"]
                step_clip += stats["clip_ratio"]
                n_pg += 1
                del cur_lp, loss

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _maybe_empty_cache(device)
        update_sec = time.monotonic() - update_t0
        step_sec = time.monotonic() - step_t0

        print(
            f"step={step}/{max_steps} batch={batch_size} k={k} gen_batch={gen_batch_size} "
            f"repeats={reward_repeats} unique_hints={n_unique_hints}/{batch_size * k} "
            f"oss_calls={n_oss_calls} em={step_em/batch_size:.0%} "
            f"groups=correct:{group_types['all_correct']},"
            f"wrong:{group_types['all_wrong']},mixed:{group_types['mixed']},"
            f"tied:{group_types['tied']},empty:{group_types['empty']} "
            f"repeat_disagree={repeat_disagreements}/{n_unique_hints} "
            f"api_errors={sum(error_counts.values())} "
            f"pg={step_pg/max(n_pg,1):.4f} kl={step_kl/max(n_pg,1):.5f} "
            f"clip={step_clip/max(n_pg,1):.3f} grad={grad_norm:.4f} "
            f"sec=gen:{gen_sec:.1f},reward:{reward_sec:.1f},"
            f"update:{update_sec:.1f},total:{step_sec:.1f}",
            flush=True,
        )
        _save_step(step)
        _maybe_empty_cache(device)

    print(
        f"skipped_groups={skipped_groups}/{total_groups} "
        f"({100*skipped_groups/max(total_groups,1):.1f}%)",
        flush=True,
    )
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"adapter -> {adapter_dir}")
    return adapter_dir


def merge_ff(cfg: Config, *, adapter_dir: Path | None = None, merged_dir: Path | None = None) -> Path:
    from peft import PeftModel

    adapter_dir = Path(adapter_dir or cfg.ff_adapter_dir)
    merged_dir = Path(merged_dir or cfg.ff_merged_dir)

    tok = AutoTokenizer.from_pretrained(cfg.router_base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.router_base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print(f"merged -> {merged_dir}")
    return merged_dir
