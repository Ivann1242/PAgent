#!/usr/bin/env python3
"""Train short-label residual judge/selector/action tasks with balanced LoRA SFT."""

from __future__ import annotations

import os

# Local safety invariant: this project may use physical GPU3 only.
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

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


def format_chat(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            messages, **kwargs, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _balanced_rows(
    rows: list[dict],
    *,
    tasks: set[str],
    seed: int,
    max_per_task: int,
    balanced_task_size: int,
) -> list[dict]:
    rng = random.Random(seed)
    raw_buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task") or "")
        if task in tasks and row.get("target") and row.get("system") and row.get("prompt"):
            raw_buckets[task].append(row)
    buckets: dict[str, list[dict]] = {}
    for task, bucket in raw_buckets.items():
        rng.shuffle(bucket)
        if max_per_task > 0:
            del bucket[max_per_task:]
        # Short-label tasks are balanced by target before tasks are balanced.
        if any(row.get("options") for row in bucket):
            by_target: dict[str, list[dict]] = defaultdict(list)
            for row in bucket:
                by_target[str(row["target"])].append(row)
            target_n = max(len(group) for group in by_target.values())
            balanced_task: list[dict] = []
            for group in by_target.values():
                rng.shuffle(group)
                balanced_task.extend(group[i % len(group)] for i in range(target_n))
            rng.shuffle(balanced_task)
            buckets[task] = balanced_task
        else:
            buckets[task] = bucket
    if not buckets:
        return []
    target_n = max(len(bucket) for bucket in buckets.values())
    if balanced_task_size > 0:
        target_n = min(target_n, balanced_task_size)
    balanced: list[dict] = []
    for task, bucket in sorted(buckets.items()):
        if not bucket:
            continue
        rng.shuffle(bucket)
        balanced.extend(
            {**bucket[i % len(bucket)], "_balanced_task": task}
            for i in range(target_n)
        )
    rng.shuffle(balanced)
    return balanced


def _validate_task_coverage(
    rows: list[dict],
    *,
    tasks: set[str],
    split: str,
    min_rows: int,
) -> None:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task") or "")].append(row)
    for task in sorted(tasks):
        task_rows = by_task.get(task, [])
        if len(task_rows) < min_rows:
            raise SystemExit(
                f"{split} task {task!r} has {len(task_rows)} rows; need {min_rows}"
            )
        labels = {str(row.get("target") or "") for row in task_rows}
        if task == "correctness" and not {"CORRECT", "INCORRECT"} <= labels:
            raise SystemExit(f"{split} correctness missing both labels")
        if task == "selection" and not {"KEEP", "REPLACE"} <= labels:
            raise SystemExit(f"{split} selection missing both labels")
        if task == "action" and len(
            labels
            & {
                "STOP",
                "VERIFY_REPAIR",
                "ALTERNATE_SOLVE",
                "TARGETED_CHECK",
            }
        ) < 2:
            raise SystemExit(f"{split} action needs at least two target actions")


class FeedbackDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def _collate(rows: list[dict]) -> list[dict]:
    return rows


def _encode(
    tokenizer,
    prompt: str,
    target: str,
    *,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    target_text = target + (tokenizer.eos_token or "")
    target_ids = tokenizer(
        target_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    room = max(max_length - len(target_ids), 1)
    if len(prompt_ids) > room:
        left = room // 2
        right = room - left
        prompt_ids = torch.cat([prompt_ids[:left], prompt_ids[-right:]])
    input_ids = torch.cat([prompt_ids, target_ids]).unsqueeze(0).to(device)
    labels = torch.cat(
        [
            torch.full_like(prompt_ids, -100),
            target_ids,
        ]
    ).unsqueeze(0).to(device)
    return input_ids, labels


def _row_loss(
    model,
    tokenizer,
    row: dict,
    device: torch.device,
    *,
    max_length: int,
) -> torch.Tensor:
    chat = format_chat(tokenizer, row["system"], row["prompt"])
    input_ids, labels = _encode(
        tokenizer,
        chat,
        str(row["target"]),
        device=device,
        max_length=max_length,
    )
    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss * float(row.get("weight") or 1.0)


def _batch_loss(
    model,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    *,
    max_length: int,
) -> torch.Tensor:
    encoded = []
    for row in rows:
        chat = format_chat(tokenizer, row["system"], row["prompt"])
        encoded.append(
            _encode(
                tokenizer,
                chat,
                str(row["target"]),
                device=device,
                max_length=max_length,
            )
        )
    max_len = max(input_ids.size(1) for input_ids, _ in encoded)
    batch_size = len(encoded)
    pad_id = tokenizer.pad_token_id
    input_batch = torch.full(
        (batch_size, max_len), pad_id, dtype=torch.long, device=device
    )
    label_batch = torch.full(
        (batch_size, max_len), -100, dtype=torch.long, device=device
    )
    attention = torch.zeros(
        (batch_size, max_len), dtype=torch.long, device=device
    )
    for index, (input_ids, labels) in enumerate(encoded):
        length = input_ids.size(1)
        input_batch[index, :length] = input_ids[0]
        label_batch[index, :length] = labels[0]
        attention[index, :length] = 1
    logits = model(input_ids=input_batch, attention_mask=attention).logits[:, :-1, :]
    targets = label_batch[:, 1:]
    mask = targets.ne(-100)
    safe_targets = targets.masked_fill(~mask, 0)
    token_loss = F.cross_entropy(
        logits.transpose(1, 2),
        safe_targets,
        reduction="none",
    )
    sample_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    weights = torch.tensor(
        [float(row.get("weight") or 1.0) for row in rows],
        dtype=sample_loss.dtype,
        device=device,
    )
    return (sample_loss * weights).mean()


@torch.no_grad()
def _label_score(
    model,
    tokenizer,
    row: dict,
    label: str,
    device: torch.device,
    *,
    max_length: int,
) -> float:
    chat = format_chat(tokenizer, row["system"], row["prompt"])
    input_ids, labels = _encode(
        tokenizer, chat, label, device=device, max_length=max_length
    )
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    safe_targets = targets.masked_fill(~mask, 0)
    token_lp = torch.gather(
        F.log_softmax(logits, dim=-1), 2, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    return float((token_lp * mask).sum() / mask.sum().clamp_min(1))


def _ece(probabilities: list[float], labels: list[int], bins: int = 10) -> float:
    if not probabilities:
        return 0.0
    total = len(probabilities)
    value = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        members = [
            i
            for i, probability in enumerate(probabilities)
            if lo <= probability < hi or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = mean(probabilities[i] for i in members)
        accuracy = mean(labels[i] for i in members)
        value += len(members) / total * abs(confidence - accuracy)
    return value


@torch.no_grad()
def _free_generation_metrics(
    model,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    *,
    max_length: int,
    max_per_task: int,
) -> dict[str, Any]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("task") or "")].append(row)
    result: dict[str, Any] = {}
    for task, bucket in sorted(buckets.items()):
        subset = bucket[:max_per_task]
        valid = correct = 0
        for row in subset:
            chat = format_chat(tokenizer, row["system"], row["prompt"])
            input_ids = tokenizer(
                chat, add_special_tokens=False, return_tensors="pt"
            ).input_ids
            input_ids = input_ids[:, -(max_length - 64) :].to(device)
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=128 if task == "diagnosis" else 12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(
                generated[0, input_ids.size(1) :], skip_special_tokens=True
            ).strip()
            if task == "diagnosis":
                try:
                    parsed = json.loads(text)
                    is_valid = isinstance(parsed, dict) and "error_type" in parsed
                except json.JSONDecodeError:
                    is_valid = False
                valid += int(is_valid)
            else:
                label = text.upper().split()[0] if text else ""
                options = [str(value) for value in row.get("options") or []]
                is_valid = label in options
                valid += int(is_valid)
                correct += int(is_valid and label == row["target"])
        result[task] = {
            "n": len(subset),
            "valid_rate": valid / len(subset) if subset else 0.0,
            "accuracy": (
                correct / len(subset)
                if subset and task != "diagnosis"
                else None
            ),
        }
    return result


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    *,
    max_length: int,
    max_per_task: int,
    generation_smoke_per_task: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("options"):
            buckets[str(row.get("task"))].append(row)
    metrics: dict[str, Any] = {}
    model.eval()
    for task, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        subset = bucket[:max_per_task] if max_per_task > 0 else bucket
        correct = 0
        by_target_total: dict[str, int] = defaultdict(int)
        by_target_correct: dict[str, int] = defaultdict(int)
        strict_total = strict_correct = 0
        regrets: list[float] = []
        correctness_probs: list[float] = []
        correctness_labels: list[int] = []
        for row in subset:
            options = [str(x) for x in row.get("options") or []]
            scores = [
                _label_score(
                    model,
                    tokenizer,
                    row,
                    option,
                    device,
                    max_length=max_length,
                )
                for option in options
            ]
            prediction = options[max(range(len(options)), key=lambda i: scores[i])]
            correct += int(prediction == row["target"])
            by_target_total[str(row["target"])] += 1
            by_target_correct[str(row["target"])] += int(
                prediction == row["target"]
            )
            if task == "selection" and not (row.get("metadata") or {}).get("tie"):
                strict_total += 1
                strict_correct += int(prediction == row["target"])
            if task == "correctness" and set(options) == {"CORRECT", "INCORRECT"}:
                logits = torch.tensor(scores, dtype=torch.float32)
                probs = torch.softmax(logits, dim=0)
                correct_index = options.index("CORRECT")
                correctness_probs.append(float(probs[correct_index]))
                correctness_labels.append(int(row["target"] == "CORRECT"))
            if task == "action":
                q_values = (row.get("metadata") or {}).get("q_values") or {}
                if q_values and prediction in q_values:
                    regrets.append(
                        max(float(v) for v in q_values.values())
                        - float(q_values[prediction])
                    )
        entry: dict[str, Any] = {
            "n": len(subset),
            "accuracy": correct / len(subset) if subset else 0.0,
            "balanced_accuracy": (
                mean(
                    by_target_correct[target] / count
                    for target, count in by_target_total.items()
                )
                if by_target_total
                else 0.0
            ),
            "by_target": {
                target: {
                    "n": count,
                    "accuracy": by_target_correct[target] / count,
                }
                for target, count in sorted(by_target_total.items())
            },
        }
        if regrets:
            entry["mean_regret"] = mean(regrets)
        if task == "selection":
            entry["strict_n"] = strict_total
            entry["strict_accuracy"] = (
                strict_correct / strict_total if strict_total else 0.0
            )
        if correctness_probs:
            entry["brier"] = mean(
                (p - y) ** 2
                for p, y in zip(correctness_probs, correctness_labels)
            )
            entry["ece"] = _ece(correctness_probs, correctness_labels)
        metrics[task] = entry
    if generation_smoke_per_task > 0:
        metrics["free_generation"] = _free_generation_metrics(
            model,
            tokenizer,
            rows,
            device,
            max_length=max_length,
            max_per_task=generation_smoke_per_task,
        )
    model.train()
    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.gpu) != "3":
        raise SystemExit("local safety: residual feedback training is restricted to GPU3")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    raw_train_rows = load_jsonl(Path(args.train_file))
    present_tasks = {
        str(row.get("task") or "") for row in raw_train_rows if row.get("target")
    }
    missing_tasks = tasks - present_tasks
    if missing_tasks:
        raise SystemExit(
            f"requested tasks missing from training data: {sorted(missing_tasks)}"
        )
    raw_val_rows = load_jsonl(Path(args.val_file))
    _validate_task_coverage(
        raw_train_rows,
        tasks=tasks,
        split="train",
        min_rows=args.min_task_rows,
    )
    _validate_task_coverage(
        raw_val_rows,
        tasks=tasks,
        split="val",
        min_rows=args.min_task_rows,
    )
    train_rows = _balanced_rows(
        raw_train_rows,
        tasks=tasks,
        seed=args.seed,
        max_per_task=args.max_per_task,
        balanced_task_size=args.balanced_task_size,
    )
    val_rows = [
        row
        for row in raw_val_rows
        if str(row.get("task") or "") in tasks
    ]
    if not train_rows:
        raise SystemExit("no training rows")
    print(
        f"residual feedback train rows={len(train_rows)} val={len(val_rows)} "
        f"tasks={sorted(tasks)} device={device}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
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
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()
    loader = DataLoader(
        FeedbackDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.01,
    )
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        losses: list[float] = []
        accum_count = 0
        for batch in loader:
            loss = _batch_loss(
                model,
                tokenizer,
                batch,
                device,
                max_length=args.max_length,
            )
            loss.backward()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
            accum_count += 1
            if accum_count == args.grad_accum:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accum_count)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
            if args.log_every and global_step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} loss={losses[-1]:.4f}",
                    flush=True,
                )
        if accum_count:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accum_count)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(
            model,
            tokenizer,
            val_rows,
            device,
            max_length=args.max_length,
            max_per_task=args.eval_max_per_task,
            generation_smoke_per_task=args.generation_smoke_per_task,
            seed=args.seed + epoch,
        )
        epoch_row = {
            "epoch": epoch,
            "loss": mean(losses) if losses else math.nan,
            "metrics": metrics,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metadata = {
        "base_model": str(args.base_model),
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "tasks": sorted(tasks),
        "n_train_balanced": len(train_rows),
        "n_val": len(val_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "history": history,
        "gpu": 3,
    }
    (adapter_dir / "feedback_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.merged_dir:
        merged_dir = Path(args.merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        (merged_dir / "feedback_meta.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"merged feedback model -> {merged_dir}", flush=True)
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--merged-dir", default="")
    p.add_argument("--base-model", type=Path, default=ROUTER_BASE)
    p.add_argument("--gpu", default="3")
    p.add_argument(
        "--tasks", default="correctness,selection,action,diagnosis"
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--max-per-task", type=int, default=0)
    p.add_argument("--balanced-task-size", type=int, default=1024)
    p.add_argument("--eval-max-per-task", type=int, default=200)
    p.add_argument("--min-task-rows", type=int, default=8)
    p.add_argument("--generation-smoke-per-task", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
