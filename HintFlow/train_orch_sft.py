#!/usr/bin/env python3
"""LoRA SFT for full HintFlow plan/review JSON completions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ROUTER_BASE  # noqa: E402
from core import load_jsonl  # noqa: E402


def _valid_json(text: str) -> bool:
    try:
        value = json.loads((text or "").strip())
        return isinstance(value, dict)
    except (TypeError, json.JSONDecodeError):
        return False


def _chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, **kwargs, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _tree_in_val(tree_id: Any, val_ratio: float, seed: int) -> bool:
    digest = hashlib.sha1(f"{seed}:{tree_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < val_ratio


def _split_rows(
    rows: list[dict], *, val_ratio: float, seed: int
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    clean = [
        row
        for row in rows
        if row.get("system")
        and row.get("prompt")
        and _valid_json(str(row.get("target") or ""))
        and row.get("task") in {"plan", "review"}
    ]
    train = [
        row
        for row in clean
        if not _tree_in_val(row.get("tree_id"), val_ratio, seed)
    ]
    val = [
        row
        for row in clean
        if _tree_in_val(row.get("tree_id"), val_ratio, seed)
    ]
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    meta = {
        "raw": len(rows),
        "clean": len(clean),
        "filtered": len(rows) - len(clean),
        "train": len(train),
        "val": len(val),
        "train_plan": sum(row["task"] == "plan" for row in train),
        "train_review": sum(row["task"] == "review" for row in train),
        "val_plan": sum(row["task"] == "plan" for row in val),
        "val_review": sum(row["task"] == "review" for row in val),
    }
    return train, val, meta


class Rows(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def _encode(
    tokenizer,
    row: dict,
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = _chat_prompt(tokenizer, row["system"], row["prompt"])
    target = str(row["target"]).strip() + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    target_ids = tokenizer(
        target, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]

    if len(target_ids) >= max_length:
        target_ids = target_ids[: max_length - 1]
        if tokenizer.eos_token_id is not None:
            target_ids[-1] = tokenizer.eos_token_id
    room = max(max_length - len(target_ids), 1)
    if len(prompt_ids) > room:
        # Preserve both the problem header and the most recent solver state.
        left = max(room // 3, 1)
        right = room - left
        prompt_ids = torch.cat([prompt_ids[:left], prompt_ids[-right:]])
    ids = torch.cat([prompt_ids, target_ids])
    labels = torch.cat(
        [torch.full_like(prompt_ids, -100), target_ids]
    )
    return ids, labels


def _batch(
    tokenizer,
    rows: list[dict],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [
        _encode(tokenizer, row, max_length=max_length) for row in rows
    ]
    max_len = max(len(ids) for ids, _ in encoded)
    pad_id = tokenizer.pad_token_id
    ids_batch = torch.full(
        (len(encoded), max_len), pad_id, dtype=torch.long
    )
    labels_batch = torch.full(
        (len(encoded), max_len), -100, dtype=torch.long
    )
    mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for i, (ids, labels) in enumerate(encoded):
        n = len(ids)
        ids_batch[i, :n] = ids
        labels_batch[i, :n] = labels
        mask[i, :n] = 1
    return (
        ids_batch.to(device),
        labels_batch.to(device),
        mask.to(device),
    )


def _loss(model, ids, labels, mask) -> torch.Tensor:
    logits = model(
        input_ids=ids, attention_mask=mask, use_cache=False
    ).logits[:, :-1]
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    safe = targets.masked_fill(~valid, 0)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe.reshape(-1),
        reduction="none",
    ).view_as(safe)
    sample_loss = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
    return sample_loss.mean()


@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    rows: list[dict],
    *,
    max_length: int,
    device: torch.device,
    limit: int,
) -> dict[str, float]:
    model.eval()
    selected = rows[:limit] if limit > 0 else rows
    by_task: dict[str, list[float]] = {"plan": [], "review": []}
    losses: list[float] = []
    for row in selected:
        ids, labels, mask = _batch(
            tokenizer, [row], max_length=max_length, device=device
        )
        value = float(_loss(model, ids, labels, mask).cpu())
        losses.append(value)
        by_task[row["task"]].append(value)
    model.train()
    return {
        "loss": mean(losses) if losses else math.nan,
        "plan_loss": mean(by_task["plan"]) if by_task["plan"] else math.nan,
        "review_loss": (
            mean(by_task["review"]) if by_task["review"] else math.nan
        ),
        "n": len(selected),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--merged-dir", default="")
    p.add_argument("--base-model", type=Path, default=ROUTER_BASE)
    p.add_argument("--gpu", default="3")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--eval-limit", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="smoke subset")
    args = p.parse_args()

    if str(args.gpu) != "3":
        raise SystemExit("local safety: HintFlow SFT is restricted to GPU3")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    raw = load_jsonl(Path(args.data_file))
    if args.limit > 0:
        raw = raw[: args.limit]
    train_rows, val_rows, split_meta = _split_rows(
        raw, val_ratio=args.val_ratio, seed=args.seed
    )
    if not train_rows or not val_rows:
        raise SystemExit(f"empty split: {split_meta}")
    print(json.dumps({"split": split_meta}, indent=2), flush=True)

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    loader = DataLoader(
        Rows(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    optimizer = torch.optim.AdamW(
        (x for x in model.parameters() if x.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-10,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    initial_val = evaluate(
        model,
        tokenizer,
        val_rows,
        max_length=args.max_length,
        device=device,
        limit=args.eval_limit,
    )
    print(json.dumps({"initial_val": initial_val}), flush=True)

    history: list[dict[str, Any]] = []
    global_micro = 0
    updates = 0
    for epoch in range(1, args.epochs + 1):
        losses: list[float] = []
        accum = 0
        for rows in loader:
            ids, labels, mask = _batch(
                tokenizer,
                rows,
                max_length=args.max_length,
                device=device,
            )
            loss = _loss(model, ids, labels, mask)
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
            global_micro += 1
            accum += 1
            if accum == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                accum = 0
            if args.log_every and global_micro % args.log_every == 0:
                print(
                    f"epoch={epoch} micro={global_micro} updates={updates} "
                    f"loss={mean(losses[-args.log_every:]):.4f}",
                    flush=True,
                )
        if accum:
            # Correct the final partial accumulation's scale.
            scale = args.grad_accum / accum
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        val = evaluate(
            model,
            tokenizer,
            val_rows,
            max_length=args.max_length,
            device=device,
            limit=args.eval_limit,
        )
        row = {
            "epoch": epoch,
            "train_loss": mean(losses),
            "val": val,
            "updates": updates,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metadata = {
        "base_model": str(args.base_model),
        "data_file": args.data_file,
        "split": split_meta,
        "initial_val": initial_val,
        "history": history,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    (adapter_dir / "sft_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    if args.merged_dir:
        merged_dir = Path(args.merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        (merged_dir / "sft_meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(f"merged -> {merged_dir}", flush=True)


if __name__ == "__main__":
    main()
