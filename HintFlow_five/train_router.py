#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from core import append_jsonl, call_llm, format_router_input, load_jsonl
from HintFlow_five.common import N_TURNS, RANKER_SYSTEM, SELECTOR_SYSTEM, build_ranker_prompt, build_router_prompt, build_selector_prompt, build_step_prompt, parse_json_object
from grpo import completion_logprobs, encode_prompt_completion, grpo_loss, group_advantages
from train_ff import _generate_k_samples

BASE = "/home/ivaning/prompt-r1r/Prompt-R1/Qwen/Qwen3-4B"


class TurnDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt = format_router_input(
            self.tokenizer,
            build_router_prompt(row["problem"], row["accepted_steps"], int(row["turn"])),
        )
        return prompt, row["label_hint"].strip()


def collate(tokenizer, batch):
    ids_list, labels_list = [], []
    eos = tokenizer.eos_token or ""
    for prompt, completion in batch:
        p = tokenizer(prompt, add_special_tokens=False).input_ids
        c = tokenizer(completion + eos, add_special_tokens=False).input_ids
        ids = torch.tensor(p + c, dtype=torch.long)
        labels = torch.tensor([-100] * len(p) + c, dtype=torch.long)
        ids_list.append(ids)
        labels_list.append(labels)
    length = max(x.numel() for x in ids_list)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(batch), length), pad, dtype=torch.long)
    labels = torch.full((len(batch), length), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), length), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(ids_list, labels_list)):
        input_ids[i, : ids.numel()] = ids
        labels[i, : lab.numel()] = lab
        attention[i, : ids.numel()] = 1
    return input_ids, labels, attention


def setup():
    local = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    return torch.device(f"cuda:{local}"), rank, world


def train(args):
    device, rank, world = setup()
    if args.batch_size % world:
        raise ValueError("global batch size must be divisible by world size")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_jsonl(Path(args.data))
    ids = sorted({r["id"] for r in rows})
    random.Random(args.seed).shuffle(ids)
    n_val = max(1, round(len(ids) * args.val_ratio))
    val_ids = set(ids[:n_val])
    train_rows = [r for r in rows if r["id"] not in val_ids]
    val_rows = [r for r in rows if r["id"] in val_ids]
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = TurnDataset(train_rows, tok)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size // world,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=lambda x: collate(tok, x),
        num_workers=2,
        pin_memory=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.gradient_checkpointing_enable()
    if world > 1:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        loss_sum = 0.0
        for step, batch in enumerate(loader, 1):
            input_ids, labels, attention = (x.to(device, non_blocking=True) for x in batch)
            loss = model(input_ids=input_ids, labels=labels, attention_mask=attention).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            if rank == 0 and step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} loss={loss_sum/step:.4f}", flush=True)
        if rank == 0:
            print(f"epoch={epoch}/{args.epochs} train_loss={loss_sum/max(len(loader),1):.4f}", flush=True)
    raw = model.module if isinstance(model, DDP) else model
    if rank == 0:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        raw.save_pretrained(out)
        tok.save_pretrained(out)
        meta = vars(args) | {"n_train_questions": len(ids) - n_val, "n_val_questions": n_val,
                             "n_train_turns": len(train_rows), "n_val_turns": len(val_rows), "world_size": world}
        (out / "sft_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def merge(args):
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"merged -> {out}")


def _api_call(client, model, prompt, max_tokens, seed):
    return call_llm(client, model, prompt, temperature=0.0, max_tokens=max_tokens, seed=seed, extra_body={"chat_template_kwargs": {"enable_thinking": False}}).strip()


def _judge_call(client, model, system, prompt, max_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (response.choices[0].message.content or "").strip()


def _completion_lps(model, tokenizer, prompt, completion, device, grad=False):
    ids, start = encode_prompt_completion(tokenizer, prompt, completion, device)
    context = torch.enable_grad() if grad else torch.inference_mode()
    with context:
        return completion_logprobs(model, ids, start)


def _sample_hint_group(model, tokenizer, prompt, *, k, max_new_tokens, temperature, device, min_unique, max_rounds):
    completions = []
    for round_index in range(max_rounds):
        completions = _generate_k_samples(
            model, tokenizer, prompt, k=k, max_new_tokens=max_new_tokens,
            temperature=temperature * (1.0 + 0.35 * round_index), device=device,
        )
        if len({text.strip() for text in completions}) >= min(min_unique, k):
            break
    return completions


def _rank_positive(client, model, problem, state, turn, positives, hints, hinted, n_turns, max_tokens, seed):
    if len(positives) <= 1:
        return [positives[0]] if positives else [], True, "single-or-empty"
    items = [(i + 1, hints[k], hinted[k]) for i, k in enumerate(positives)]
    raw = _judge_call(client, model, RANKER_SYSTEM, build_ranker_prompt(problem, state, turn, items, n_turns), max_tokens)
    try:
        ranking = parse_json_object(raw).get("ranking", [])
        ids = [int(x) for x in ranking]
        expected = list(range(1, len(positives) + 1))
        if len(ids) != len(expected) or sorted(ids) != expected:
            raise ValueError("ranking must be a permutation")
        return [positives[i - 1] for i in ids], True, raw
    except Exception:
        return list(positives), False, raw


def train_grpo(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_jsonl(Path(args.data))
    if args.k < 2:
        raise SystemExit("GRPO requires --k >= 2")
    if args.turns < 1:
        raise SystemExit("--turns must be positive")
    if not rows:
        raise SystemExit("empty training data")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    policy = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, trust_remote_code=True).to(device)
    policy = get_peft_model(policy, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    reference = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, trust_remote_code=True).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    policy.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    solver_urls = [x.strip() for x in args.solver_urls.split(",") if x.strip()]
    if not solver_urls:
        raise SystemExit("--solver-urls is empty")
    solvers = [OpenAI(base_url=url, api_key="EMPTY", timeout=3600, max_retries=0) for url in solver_urls]
    judge = OpenAI(base_url=args.selector_url, api_key="EMPTY", timeout=3600, max_retries=0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rollout_log = out / "rollouts.jsonl"
    cursor = 0
    for update in range(1, args.max_steps + 1):
        started = time.monotonic()
        groups = []
        batch = [rows[(cursor + i) % len(rows)] for i in range(args.batch_size)]
        cursor = (cursor + args.batch_size) % len(rows)
        policy.eval()
        for row_i, row in enumerate(batch):
            state = []
            solver = solvers[(cursor + row_i) % len(solvers)]
            for turn in range(1, args.turns + 1):
                seed0 = args.seed + update * 100000 + row_i * 1000 + turn * 20
                router_text = build_router_prompt(row["problem"], state, turn, args.turns)
                prompt = format_router_input(tokenizer, router_text)
                completions = _sample_hint_group(policy, tokenizer, prompt, k=args.k, max_new_tokens=args.router_max_tokens, temperature=args.router_temperature, device=device, min_unique=args.min_unique, max_rounds=args.max_sample_rounds)
                hints = [text.strip() for text in completions]
                old_lps = [_completion_lps(policy, tokenizer, prompt, text, device).detach() for text in completions]
                ref_lps = [_completion_lps(reference, tokenizer, prompt, text, device).detach() for text in completions]
                baseline = _api_call(solver, args.solver_model, build_step_prompt(row["problem"], state, turn, n_turns=args.turns), args.solver_max_tokens, seed0)
                hint_to_indices = {}
                for k, hint in enumerate(hints):
                    hint_to_indices.setdefault(hint, []).append(k)
                representatives = [indices[0] for indices in hint_to_indices.values()]
                def evaluate_hint(k):
                    hint = hints[k]
                    candidate = _api_call(solver, args.solver_model, build_step_prompt(row["problem"], state, turn, hint, args.turns), args.solver_max_tokens, seed0 + k + 1)
                    raw = _judge_call(judge, args.selector_model, SELECTOR_SYSTEM, build_selector_prompt(row["problem"], state, turn, baseline, candidate, args.turns), args.selector_max_tokens)
                    try:
                        obj = parse_json_object(raw)
                        decision = str(obj.get("decision", "BASELINE")).upper().strip()
                        parse_ok = decision in {"BASELINE", "HINTED"}
                        if not parse_ok:
                            decision = "BASELINE"
                    except Exception:
                        decision, parse_ok = "BASELINE", False
                    return k, candidate, decision, {"decision": decision, "parse_ok": parse_ok, "raw": raw}
                with ThreadPoolExecutor(max_workers=len(representatives)) as pool:
                    unique_evaluated = list(pool.map(evaluate_hint, representatives))
                by_rep = {item[0]: item[1:] for item in unique_evaluated}
                hinted = [""] * args.k
                decisions = ["BASELINE"] * args.k
                selector_records = [{} for _ in range(args.k)]
                for indices in hint_to_indices.values():
                    candidate, decision, selector_record = by_rep[indices[0]]
                    for k in indices:
                        hinted[k] = candidate
                        decisions[k] = decision
                        selector_records[k] = selector_record
                positives = [k for k in representatives if decisions[k] == "HINTED"]
                ranked, rank_ok, rank_raw = _rank_positive(judge, args.selector_model, row["problem"], state, turn, positives, hints, hinted, args.turns, args.ranker_max_tokens, seed0 + args.k * 2 + 1)
                rewards = [0.0] * args.k
                if ranked:
                    count = len(ranked)
                    for rank, representative in enumerate(ranked):
                        quality = 1.0 if count == 1 else (count - rank - 1) / (count - 1)
                        reward = 1.0 + args.rank_weight * quality
                        for k in hint_to_indices[hints[representative]]:
                            rewards[k] = reward
                    accepted = hinted[ranked[0]]
                    accepted_source = f"hinted:{ranked[0]}"
                else:
                    accepted = baseline
                    accepted_source = "baseline"
                record = {"update": update, "id": row.get("id"), "turn": turn, "problem": row["problem"], "accepted_before": list(state), "baseline": baseline, "completions": completions, "hints": hints, "n_unique_hints": len(hint_to_indices), "hinted": hinted, "selectors": selector_records, "positive_indices": positives, "ranking": ranked, "rank_parse_ok": rank_ok, "rank_raw": rank_raw, "rewards": rewards, "accepted_source": accepted_source, "accepted": accepted}
                append_jsonl(rollout_log, record)
                groups.append({"prompt": prompt, "completions": completions, "old_lps": old_lps, "ref_lps": ref_lps, "rewards": rewards})
                state.append(accepted)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        signal_groups = 0
        total_loss = 0.0
        total_items = 0
        scale = 1.0 / max(len(groups) * args.k, 1)
        for group in groups:
            advantages, _, _, has_signal = group_advantages(group["rewards"], method="grpo")
            signal_groups += int(has_signal)
            for completion, old_lp, ref_lp, advantage in zip(group["completions"], group["old_lps"], group["ref_lps"], advantages.tolist()):
                current_lp = _completion_lps(policy, tokenizer, group["prompt"], completion, device, grad=True)
                loss, _ = grpo_loss(current_lp, old_lp, ref_lp, advantage, clip=args.clip, beta=args.kl_beta)
                (loss * scale).backward()
                total_loss += float(loss.detach())
                total_items += 1
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
        optimizer.step()
        policy.save_pretrained(out)
        tokenizer.save_pretrained(out)
        state_meta = {"update": update, "cursor": cursor, "base": args.base, "turns": args.turns, "k": args.k, "batch_size": args.batch_size, "rank_weight": args.rank_weight, "signal_groups": signal_groups, "groups": len(groups)}
        (out / "grpo_state.json").write_text(json.dumps(state_meta, indent=2) + "\n")
        print(f"update={update}/{args.max_steps} groups={len(groups)} signal={signal_groups} loss={total_loss/max(total_items,1):.4f} grad={grad_norm:.4f} sec={time.monotonic()-started:.1f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="checkpoints/hintflow_five_data/sft_turns.jsonl")
    t.add_argument("--base", default=BASE)
    t.add_argument("--out", default="checkpoints/hintflow_five_router_adapter")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--val-ratio", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=42)
    m = sub.add_parser("merge")
    m.add_argument("--base", default=BASE)
    m.add_argument("--adapter", default="checkpoints/hintflow_five_router_adapter")
    m.add_argument("--out", default="checkpoints/hintflow_five_router_merged")
    g = sub.add_parser("grpo", help="on-policy multi-step HF1-style GRPO from the bare 4B base")
    g.add_argument("--data", default="checkpoints/hintflow_five_data/teacher_questions.jsonl")
    g.add_argument("--base", default=BASE)
    g.add_argument("--out", default="checkpoints/hintflow_five_grpo_adapter")
    g.add_argument("--gpu", default="1")
    g.add_argument("--solver-urls", default="http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1")
    g.add_argument("--solver-model", default="qwen3-14b")
    g.add_argument("--selector-url", default="http://127.0.0.1:8086/v1")
    g.add_argument("--selector-model", default="qwen3-4b-blind-ff-17k")
    g.add_argument("--turns", type=int, default=5)
    g.add_argument("--k", type=int, default=4)
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument("--max-steps", type=int, default=100)
    g.add_argument("--lr", type=float, default=2e-6)
    g.add_argument("--router-temperature", type=float, default=0.8)
    g.add_argument("--min-unique", type=int, default=2)
    g.add_argument("--max-sample-rounds", type=int, default=3)
    g.add_argument("--router-max-tokens", type=int, default=192)
    g.add_argument("--solver-max-tokens", type=int, default=2048)
    g.add_argument("--selector-max-tokens", type=int, default=192)
    g.add_argument("--ranker-max-tokens", type=int, default=384)
    g.add_argument("--rank-weight", type=float, default=1.0)
    g.add_argument("--clip", type=float, default=0.2)
    g.add_argument("--kl-beta", type=float, default=0.02)
    g.add_argument("--seed", type=int, default=20260822)
    args = p.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "merge":
        merge(args)
    else:
        train_grpo(args)


if __name__ == "__main__":
    main()
