#!/usr/bin/env python3
"""Offline DPO for HintFlow orchestrator from tree-exported pairs.

Supports single-GPU and torchrun multi-GPU (DDP). Reference logprobs can be
cached to disk so restarts skip the slow precompute.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ROUTER_BASE  # noqa: E402
from core import load_jsonl  # noqa: E402
from dpo_train import (  # noqa: E402
    _collate,
    _pairwise_accuracy,
    _seq_logprob,
    _seq_logprob_no_adapter,
    dpo_loss,
)


def format_orch_chat(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            messages, **kwargs, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_main() -> bool:
    return _rank() == 0


def _barrier() -> None:
    if _is_dist():
        dist.barrier()


def _unwrap(model):
    return model.module if isinstance(model, DDP) else model


@torch.no_grad()
def precompute_ref_logps(model, tokenizer, pairs: list[dict], device) -> list[dict]:
    """Compute ref logps for a shard of pairs (used by each rank)."""
    cache: dict[tuple[str, str], float] = {}

    def _ref(chat_prompt: str, completion: str) -> float:
        key = (chat_prompt, completion)
        if key not in cache:
            cache[key] = float(
                _seq_logprob_no_adapter(
                    model, tokenizer, chat_prompt, completion, device,
                ).cpu()
            )
        return cache[key]

    enriched = []
    for i, row in enumerate(pairs):
        chat_prompt = format_orch_chat(tokenizer, row["system"], row["prompt"])
        enriched.append({
            **row,
            "chat_prompt": chat_prompt,
            "ref_chosen_logp": _ref(chat_prompt, row["chosen"]),
            "ref_rejected_logp": _ref(chat_prompt, row["rejected"]),
        })
        if _is_main() and (i + 1) % 100 == 0:
            print(f"  ref logp {i+1}/{len(pairs)} (rank0 shard)", flush=True)
    return enriched


def _save_ref_cache(path: Path, train_pairs: list[dict], val_pairs: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"meta": meta, "train": train_pairs, "val": val_pairs}
    torch.save(payload, tmp)
    tmp.replace(path)
    if _is_main():
        print(f"ref cache -> {path} train={len(train_pairs)} val={len(val_pairs)}", flush=True)


def _load_ref_cache(path: Path, meta: dict) -> tuple[list[dict], list[dict]] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    cached_meta = payload.get("meta") or {}
    # require matching key fields
    for k in ("pairs_file", "n_pairs", "seed", "val_ratio", "max_completion_chars"):
        if cached_meta.get(k) != meta.get(k):
            if _is_main():
                print(f"ref cache miss on {k}: {cached_meta.get(k)!r} != {meta.get(k)!r}", flush=True)
            return None
    train_pairs, val_pairs = payload["train"], payload["val"]
    if _is_main():
        print(
            f"loaded ref cache {path} train={len(train_pairs)} val={len(val_pairs)}",
            flush=True,
        )
    return train_pairs, val_pairs


def _prepare_pairs(
    pairs_file: Path,
    *,
    val_ratio: float,
    seed: int,
    max_completion_chars: int,
) -> tuple[list[dict], list[dict], dict]:
    pairs = load_jsonl(pairs_file)
    cleaned = []
    for r in pairs:
        c, j = (r.get("chosen") or "").strip(), (r.get("rejected") or "").strip()
        if not c or not j or not (r.get("system") and r.get("prompt")):
            continue
        if max_completion_chars > 0:
            c = c[:max_completion_chars]
            j = j[:max_completion_chars]
        cleaned.append({**r, "chosen": c, "rejected": j})
    pairs = cleaned
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    meta = {
        "pairs_file": str(pairs_file.resolve()),
        "n_pairs": len(pairs),
        "n_train": len(train_pairs),
        "n_val": n_val,
        "seed": seed,
        "val_ratio": val_ratio,
        "max_completion_chars": max_completion_chars,
    }
    return train_pairs, val_pairs, meta


class PairDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class CudaMemFence:
    """Occupy leftover free VRAM so co-tenants cannot grow into it.

    Modes:
      - sticky (default): claim once after model load and hold for the whole run,
        leaving ``keep_free_mb`` for activations. No release during steps.
      - pulsed: release around compute and re-claim between steps (weaker against
        co-tenants that allocate during our long forwards).
    """

    def __init__(
        self,
        device: torch.device,
        *,
        keep_free_mb: int = 20480,
        enabled: bool = True,
        sticky: bool = True,
    ):
        self.device = torch.device(device)
        self.keep_free = max(0, int(keep_free_mb)) * 1024 * 1024
        self.enabled = enabled and self.device.type == "cuda"
        self.sticky = sticky
        self._buf: torch.Tensor | None = None
        self._last_claim_mb = 0.0

    def release(self) -> None:
        self._buf = None
        if self.enabled:
            torch.cuda.empty_cache()

    def claim(self) -> float:
        if not self.enabled:
            return 0.0
        self.release()
        free, _total = torch.cuda.mem_get_info(self.device)
        take = int(free) - self.keep_free
        if take < 64 * 1024 * 1024:
            self._last_claim_mb = 0.0
            return 0.0
        for frac in (1.0, 0.95, 0.9, 0.8, 0.7):
            n = int(take * frac)
            if n < 64 * 1024 * 1024:
                break
            try:
                self._buf = torch.empty(n, dtype=torch.uint8, device=self.device)
                self._last_claim_mb = n / (1024 * 1024)
                return self._last_claim_mb
            except torch.cuda.OutOfMemoryError:
                self._buf = None
                torch.cuda.empty_cache()
        self._last_claim_mb = 0.0
        return 0.0

    def __enter__(self):
        # sticky: do not release — activations use keep_free headroom
        if not self.sticky:
            self.release()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.sticky:
            self.claim()
        # On OOM during sticky mode, drop fence so a retry can proceed.
        if exc_type is not None and issubclass(exc_type, torch.cuda.OutOfMemoryError):
            self.release()
        return False


def _build_model(base_model: Path, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()
    return model


def _distributed_ref_precompute(
    model,
    tokenizer,
    train_pairs: list[dict],
    val_pairs: list[dict],
    device,
    ref_cache: Path | None,
    meta: dict,
) -> tuple[list[dict], list[dict]]:
    """Each rank computes a shard; gather on all ranks; optionally save cache."""
    all_pairs = train_pairs + val_pairs
    n_train = len(train_pairs)
    world, rank = _world(), _rank()
    shard = all_pairs[rank::world]
    if _is_main():
        print(
            f"precomputing reference logprobs on {world} GPU(s); "
            f"rank0 shard={len(shard)}/{len(all_pairs)}",
            flush=True,
        )
    # disable adapter for ref; peft model supports disable_adapter
    local = precompute_ref_logps(_unwrap(model), tokenizer, shard, device)

    if world == 1:
        enriched = local
    else:
        gathered: list[list[dict] | None] = [None] * world
        dist.all_gather_object(gathered, local)
        # rebuild in original order: rank r owns indices r, r+world, ...
        enriched = [None] * len(all_pairs)  # type: ignore[list-item]
        for r, part in enumerate(gathered):
            assert part is not None
            for j, row in enumerate(part):
                enriched[r + j * world] = row
        assert all(x is not None for x in enriched)
    train_out, val_out = enriched[:n_train], enriched[n_train:]
    if ref_cache is not None and _is_main():
        _save_ref_cache(ref_cache, train_out, val_out, meta)
    _barrier()
    # non-main ranks reload from cache for identical objects (or re-gather already have)
    if ref_cache is not None and not _is_main():
        loaded = _load_ref_cache(ref_cache, meta)
        if loaded is not None:
            return loaded
    return train_out, val_out


def train(
    *,
    pairs_file: Path,
    adapter_dir: Path,
    base_model: Path,
    gpu: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    beta: float,
    val_ratio: float,
    seed: int,
    max_completion_chars: int,
    ref_cache: Path | None,
    log_every: int,
    mem_fence_mb: int,
) -> Path:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # torchrun sets LOCAL_RANK; otherwise bind via CUDA_VISIBLE_DEVICES=gpu.
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        if not _is_dist():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu.split(",")[0].strip()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    world = _world()
    rank = _rank()
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    train_pairs, val_pairs, meta = _prepare_pairs(
        pairs_file,
        val_ratio=val_ratio,
        seed=seed,
        max_completion_chars=max_completion_chars,
    )
    if _is_main():
        print(
            f"pairs={meta['n_pairs']} train={meta['n_train']} val={meta['n_val']} "
            f"beta={beta} lr={lr} epochs={epochs} world={world} "
            f"batch={batch_size} accum={grad_accum} "
            f"eff_batch={batch_size * world * grad_accum} gpu={gpu}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(base_model, device)

    # ref cache / precompute (before DDP wrap — adapter disable is cleaner)
    loaded = _load_ref_cache(ref_cache, meta) if ref_cache else None
    if loaded is not None:
        train_pairs, val_pairs = loaded
        _barrier()
    else:
        train_pairs, val_pairs = _distributed_ref_precompute(
            model, tokenizer, train_pairs, val_pairs, device, ref_cache, meta,
        )

    if world > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    sampler = (
        DistributedSampler(train_pairs, num_replicas=world, rank=rank, shuffle=True, seed=seed)
        if world > 1
        else None
    )
    loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=_collate,
        drop_last=(world > 1),
    )
    optimizer = torch.optim.AdamW(_unwrap(model).parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    fence = CudaMemFence(
        device, keep_free_mb=mem_fence_mb, enabled=mem_fence_mb > 0, sticky=True,
    )
    claimed = fence.claim()
    if _is_main():
        print(
            f"mem fence(sticky): keep_free={mem_fence_mb}MB "
            f"rank0_claimed={claimed:.0f}MB (held for whole run)",
            flush=True,
        )
    _barrier()

    global_step = 0
    history = []
    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            def _step() -> torch.Tensor:
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
                loss_local = loss_sum / max(len(batch), 1)
                (loss_local / grad_accum).backward()
                return loss_local

            try:
                with fence:
                    loss = _step()
            except torch.cuda.OutOfMemoryError:
                # sticky fence too tight for this sample — drop it and retry once
                fence.release()
                torch.cuda.empty_cache()
                if _is_main():
                    print("  OOM under sticky fence; released and retrying once", flush=True)
                loss = _step()
                fence.claim()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
            global_step += 1

            if global_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(_unwrap(model).parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if _is_main() and log_every > 0 and global_step % log_every == 0:
                print(
                    f"  epoch={epoch} step={global_step} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"fence={fence._last_claim_mb:.0f}MB",
                    flush=True,
                )

        if global_step % grad_accum != 0:
            with fence:
                torch.nn.utils.clip_grad_norm_(_unwrap(model).parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # average train loss across ranks
        loss_tensor = torch.tensor([epoch_loss, float(n_batches)], device=device)
        if world > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = float(loss_tensor[0] / max(loss_tensor[1].item(), 1.0))

        # val on rank0 only (full set) to keep metric comparable
        if _is_main():
            with fence:
                val_acc = _pairwise_accuracy(_unwrap(model), tokenizer, val_pairs, device)
            print(
                f"epoch={epoch}/{epochs} dpo_loss={train_loss:.4f} val_pair_acc={val_acc:.1%}",
                flush=True,
            )
            history.append({"epoch": epoch, "loss": train_loss, "val_pair_acc": val_acc})
        else:
            pass
        _barrier()

    if _is_main():
        adapter_dir.mkdir(parents=True, exist_ok=True)
        _unwrap(model).save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        meta_out = {
            "pairs_file": str(pairs_file),
            "n_train": len(train_pairs),
            "n_val": len(val_pairs),
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "world_size": world,
            "effective_batch": batch_size * world * grad_accum,
            "lr": lr,
            "beta": beta,
            "history": history,
            "base_model": str(base_model),
            "ref_cache": str(ref_cache) if ref_cache else None,
        }
        (adapter_dir / "dpo_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
        print(f"hintflow dpo adapter -> {adapter_dir}", flush=True)
    _barrier()
    return adapter_dir


def merge(*, adapter_dir: Path, merged_dir: Path, base_model: Path) -> Path:
    tok = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print(f"hintflow dpo merged -> {merged_dir}", flush=True)
    return merged_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs-file",
        default=str(_ROOT / "checkpoints" / "hintflow_trees_2k" / "dpo_pairs.jsonl"),
    )
    p.add_argument(
        "--adapter-dir",
        default=str(_ROOT / "checkpoints" / "hintflow_dpo_adapter"),
    )
    p.add_argument(
        "--merged-dir",
        default=str(_ROOT / "checkpoints" / "hintflow_dpo_merged"),
    )
    p.add_argument("--base-model", default=str(ROUTER_BASE))
    p.add_argument(
        "--gpu",
        default="0",
        help="CUDA_VISIBLE_DEVICES for single-process; torchrun overrides via LOCAL_RANK",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-completion-chars", type=int, default=4096)
    p.add_argument(
        "--ref-cache",
        default="",
        help="path to save/load precomputed ref logps (.pt); empty disables",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--mem-fence-mb",
        type=int,
        default=20480,
        help="sticky fence: leave this much free MB for activations; claim the rest "
             "for the whole run so co-tenants cannot grow. 0 disables.",
    )
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-merge", action="store_true")
    args = p.parse_args()

    if args.merge_only:
        merge(
            adapter_dir=Path(args.adapter_dir),
            merged_dir=Path(args.merged_dir),
            base_model=Path(args.base_model),
        )
        return

    # If launched without torchrun but --gpu has multiple ids, re-exec via torchrun.
    local_rank = os.environ.get("LOCAL_RANK")
    gpu_ids = [x.strip() for x in str(args.gpu).split(",") if x.strip() != ""]
    if local_rank is None and len(gpu_ids) > 1:
        import subprocess

        nproc = len(gpu_ids)
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--standalone",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        print(f"re-exec torchrun nproc={nproc} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
        raise SystemExit(subprocess.call(cmd, env=env))

    ref_cache = Path(args.ref_cache) if args.ref_cache else None
    train(
        pairs_file=Path(args.pairs_file),
        adapter_dir=Path(args.adapter_dir),
        base_model=Path(args.base_model),
        gpu=args.gpu if local_rank is None else os.environ.get("CUDA_VISIBLE_DEVICES", args.gpu),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        beta=args.beta,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_completion_chars=args.max_completion_chars,
        ref_cache=ref_cache,
        log_every=args.log_every,
        mem_fence_mb=args.mem_fence_mb,
    )

    # only rank0 merges; others exit after barrier inside train
    if (not args.no_merge) and _is_main():
        merge(
            adapter_dir=Path(args.adapter_dir),
            merged_dir=Path(args.merged_dir),
            base_model=Path(args.base_model),
        )

    if _is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
