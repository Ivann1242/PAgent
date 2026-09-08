#!/usr/bin/env python3
"""OpenAI-compatible HF Transformers server with near-deterministic greedy decode.

Drop-in replacement for vLLM reward/eval endpoints used by PAgent
(OpenAI client → /v1/chat/completions).

Determinism knobs:
  - do_sample=False when temperature<=0
  - batch size 1 (global lock serializes generates)
  - torch deterministic flags + CUBLAS workspace
  - optional request seed

Much slower than vLLM; use when SFT/GRPO reward noise from non-deterministic
serving matters more than throughput.
"""
from __future__ import annotations

import argparse
import os
import threading
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


def _enable_deterministic() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Some ops may still error; fall back rather than crash.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # noqa: BLE001
            pass


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 8192
    top_p: float = 1.0
    seed: int | None = None
    # accepted but ignored (HF path is always single-seq greedy/sample)
    n: int = 1
    stream: bool = False
    extra_body: dict[str, Any] | None = None


def build_app(model_path: str, served_name: str, dtype: str) -> FastAPI:
    _enable_deterministic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    print(f"loading {model_path} on {device} dtype={torch_dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    app = FastAPI(title="HF deterministic OpenAI-compatible server")
    lock = threading.Lock()
    created = int(time.time())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": served_name}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "local-hf",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(400, "stream not supported")
        if req.n != 1:
            raise HTTPException(400, "only n=1 supported")

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            # Fallback: concatenate roles
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        do_sample = float(req.temperature) > 0.0
        seed = req.seed
        if seed is None and isinstance(req.extra_body, dict):
            seed = req.extra_body.get("seed")

        with lock:
            if seed is not None:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen_kwargs: dict[str, Any] = dict(
                **inputs,
                max_new_tokens=int(req.max_tokens),
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = max(float(req.temperature), 1e-5)
                gen_kwargs["top_p"] = float(req.top_p)
            else:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():
                out = model.generate(**gen_kwargs)

            new_tokens = out[0, inputs["input_ids"].shape[-1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(new_tokens.shape[-1]),
                "total_tokens": int(inputs["input_ids"].shape[-1] + new_tokens.shape[-1]),
            },
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default="gpt-oss-20b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8006)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    args = ap.parse_args()
    app = build_app(args.model, args.served_model_name, args.dtype)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
