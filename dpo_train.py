"""Standard offline DPO for the action router (Rafailov et al., 2023)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from core import format_router_input, load_jsonl
from dpo_data import build_dpo_pairs
from grpo import completion_logprobs, encode_prompt_completion


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Standard DPO: -log sigmoid(beta * ((pi_w-ref_w) - (pi_l-ref_l)))."""
    logits = beta * (
        (policy_chosen_lp - ref_chosen_lp) - (policy_rejected_lp - ref_rejected_lp)
    )
    return (-F.logsigmoid(logits)).mean()


def _seq_logprob(model, tokenizer, prompt: str, completion: str, device, *, grad: bool = False):
    input_ids, start = encode_prompt_completion(tokenizer, prompt, completion, device)
    ctx = torch.enable_grad() if grad else torch.inference_mode()
    with ctx:
        token_lps = completion_logprobs(model, input_ids, start)
        return token_lps.sum()


def _seq_logprob_no_adapter(model, tokenizer, prompt: str, completion: str, device):
    with torch.inference_mode(), model.disable_adapter():
        return _seq_logprob(model, tokenizer, prompt, completion, device, grad=False)


@torch.no_grad()
def _precompute_ref_logps(model, tokenizer, pairs: list[dict], device) -> list[dict]:
    cache: dict[tuple[str, str], float] = {}

    def _ref(prompt: str, completion: str) -> float:
        key = (prompt, completion)
        if key not in cache:
            cache[key] = float(_seq_logprob_no_adapter(
                model, tokenizer, prompt, completion, device,
            ).cpu())
        return cache[key]

    enriched = []
    for row in pairs:
        chat_prompt = format_router_input(tokenizer, row["prompt"])
        enriched.append({
            **row,
            "chat_prompt": chat_prompt,
            "ref_chosen_logp": _ref(chat_prompt, row["chosen"]),
            "ref_rejected_logp": _ref(chat_prompt, row["rejected"]),
        })
    return enriched


class DPOPairDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def _collate(batch: list[dict]) -> list[dict]:
    return batch


@torch.no_grad()
def _pairwise_accuracy(model, tokenizer, rows: list[dict], device) -> float:
    model.eval()
    correct = 0
    for row in rows:
        chat_prompt = row["chat_prompt"]
        lp_w = float(_seq_logprob(model, tokenizer, chat_prompt, row["chosen"], device).cpu())
        lp_l = float(_seq_logprob(model, tokenizer, chat_prompt, row["rejected"], device).cpu())
        correct += int(lp_w > lp_l)
    model.train()
    return correct / max(len(rows), 1)


def train_dpo(
    cfg: Config,
    *,
    pairs_file: Path | None = None,
    rollouts_file: Path | None = None,
    adapter_dir: Path | None = None,
    gpu: str = "3",
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 5e-7,
    beta: float = 0.1,
    val_ratio: float = 0.05,
    seed: int = 42,
    rebuild_pairs: bool = False,
) -> Path:
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    rollouts_file = Path(rollouts_file or cfg.ckpt_dir / "label_2048" / "rollouts.jsonl")
    pairs_file = Path(pairs_file or cfg.ckpt_dir / "label_2048" / "dpo_pairs.jsonl")
    adapter_dir = Path(adapter_dir or cfg.ckpt_dir / "dpo_adapter")

    if rebuild_pairs or not pairs_file.exists():
        build_dpo_pairs(rollouts_file, pairs_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    pairs = load_jsonl(pairs_file)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"pairs={len(pairs)} train={len(train_pairs)} val={n_val} beta={beta}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.router_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print("precomputing reference logprobs ...")
    train_pairs = _precompute_ref_logps(model, tokenizer, train_pairs, device)
    val_pairs = _precompute_ref_logps(model, tokenizer, val_pairs, device)

    loader = DataLoader(
        DPOPairDataset(train_pairs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            loss_sum = torch.tensor(0.0, device=device)
            for row in batch:
                lp_w = _seq_logprob(
                    model, tokenizer, row["chat_prompt"], row["chosen"], device, grad=True,
                )
                lp_l = _seq_logprob(
                    model, tokenizer, row["chat_prompt"], row["rejected"], device, grad=True,
                )
                ref_w = torch.tensor(row["ref_chosen_logp"], device=device, dtype=lp_w.dtype)
                ref_l = torch.tensor(row["ref_rejected_logp"], device=device, dtype=lp_l.dtype)
                loss_sum = loss_sum + dpo_loss(lp_w, lp_l, ref_w, ref_l, beta=beta)

            loss = loss_sum / max(len(batch), 1)
            (loss / grad_accum).backward()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
            global_step += 1

            if global_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if global_step % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = epoch_loss / max(n_batches, 1)
        val_acc = _pairwise_accuracy(model, tokenizer, val_pairs, device)
        print(
            f"epoch={epoch}/{epochs} dpo_loss={train_loss:.4f} val_pair_acc={val_acc:.1%}",
            flush=True,
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    meta = {
        "pairs_file": str(pairs_file),
        "n_train": len(train_pairs),
        "n_val": n_val,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "beta": beta,
        "loss": "standard_dpo",
    }
    (adapter_dir / "dpo_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"dpo adapter -> {adapter_dir}")
    return adapter_dir


def merge_dpo(cfg: Config, *, adapter_dir: Path | None = None, merged_dir: Path | None = None) -> Path:
    from peft import PeftModel

    adapter_dir = Path(adapter_dir or cfg.ckpt_dir / "dpo_adapter")
    merged_dir = Path(merged_dir or cfg.ckpt_dir / "dpo_merged")

    tok = AutoTokenizer.from_pretrained(cfg.router_base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.router_base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print(f"dpo merged -> {merged_dir}")
    return merged_dir
