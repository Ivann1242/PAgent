"""Supervised fine-tuning for free-form prompt optimizer."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from core import build_optimizer_prompt, format_router_input, load_jsonl, normalize_answer, parse_optimizer_output


class OptimizerSFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer):
        self.samples = []
        for row in rows:
            chat_prompt = format_router_input(tokenizer, build_optimizer_prompt(row["problem"]))
            completion = row["label_hint"].strip() or "(no hint)"
            self.samples.append({
                "prompt": chat_prompt,
                "completion": completion,
                "label_hint": row["label_hint"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _collate(tokenizer, batch: list[dict], device: torch.device):
    input_ids_list = []
    labels_list = []
    for item in batch:
        p_ids = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False).input_ids[0]
        c_ids = tokenizer(item["completion"], return_tensors="pt", add_special_tokens=False).input_ids[0]
        ids = torch.cat([p_ids, c_ids])
        labels = ids.clone()
        labels[: p_ids.shape[0]] = -100
        input_ids_list.append(ids)
        labels_list.append(labels)

    max_len = max(x.shape[0] for x in input_ids_list)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : ids.shape[0]] = ids
        labels[i, : lab.shape[0]] = lab
        attn[i, : ids.shape[0]] = 1
    return input_ids.to(device), labels.to(device), attn.to(device)


@torch.no_grad()
def _eval_hint_match(model, tokenizer, rows: list[dict], device: torch.device) -> tuple[float, float]:
    model.eval()
    exact = 0
    nonempty = 0
    for row in rows:
        chat_prompt = format_router_input(tokenizer, build_optimizer_prompt(row["problem"]))
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        hint, ok = parse_optimizer_output(gen)
        nonempty += int(bool(hint.strip()) and hint.strip() != "(no hint)")
        target = row["label_hint"].strip() or "(no hint)"
        if normalize_answer(hint) == normalize_answer(target):
            exact += 1
    model.train()
    n = max(len(rows), 1)
    return exact / n, nonempty / n


def train_sft_ff(
    cfg: Config,
    *,
    labels_file: Path | None = None,
    adapter_dir: Path | None = None,
    gpu: str = "1",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    labels_file = Path(labels_file or cfg.ckpt_dir / "label_2048" / "ff_labels.jsonl")
    adapter_dir = Path(adapter_dir or cfg.ff_sft_adapter_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    rows = load_jsonl(labels_file)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"ff labels={len(rows)} train={len(train_rows)} val={n_val} from {labels_file}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.router_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = OptimizerSFTDataset(train_rows, tokenizer)
    val_ds = OptimizerSFTDataset(val_rows, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.router_base,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: _collate(tokenizer, b, device),
    )

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for input_ids, labels, attn in loader:
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        val_exact, val_nonempty = _eval_hint_match(model, tokenizer, val_rows, device)
        print(
            f"epoch={epoch}/{epochs} loss={train_loss:.4f} "
            f"val_hint_exact={val_exact:.1%} val_nonempty={val_nonempty:.1%}",
            flush=True,
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    meta = {
        "labels_file": str(labels_file),
        "n_train": len(train_rows),
        "n_val": n_val,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "mode": "freeform_sft",
    }
    (adapter_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"ff sft adapter -> {adapter_dir}")
    return adapter_dir


def merge_sft_ff(cfg: Config, *, adapter_dir: Path | None = None, merged_dir: Path | None = None) -> Path:
    from peft import PeftModel

    adapter_dir = Path(adapter_dir or cfg.ff_sft_adapter_dir)
    merged_dir = Path(merged_dir or cfg.ff_sft_merged_dir)

    tok = AutoTokenizer.from_pretrained(cfg.router_base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.router_base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print(f"ff sft merged -> {merged_dir}")
    return merged_dir
