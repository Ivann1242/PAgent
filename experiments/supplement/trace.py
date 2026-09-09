"""Persist requests BEFORE sending; no hidden API retries or lost partial text."""
from __future__ import annotations

import threading
import time
import uuid
from types import SimpleNamespace
from pathlib import Path

from .io import append


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, cap: int):
        if cap <= 0:
            raise ValueError("Budget must be positive")
        self.cap = cap
        self.reserved = 0

    def reserve(self, tokens: int):
        if tokens <= 0 or self.reserved + tokens > self.cap:
            raise BudgetExceeded(f"Requested {tokens}; remaining {self.cap - self.reserved}")
        self.reserved += tokens


class TracedClient:
    def __init__(self, client, path: Path, *, role: str, ledger: Ledger | None = None):
        self.client = client
        self.path = path
        self.role = role
        self.ledger = ledger
        self.lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        call_id = uuid.uuid4().hex
        if self.ledger:
            self.ledger.reserve(int(kwargs["max_tokens"]))
        with self.lock:
            append(self.path, {"event": "request", "call_id": call_id,
                               "role": self.role, "request": kwargs, "time": time.time()})
        started = time.monotonic()
        record = {"event": "interrupted", "usage": None}
        try:
            response = self.client.chat.completions.create(**kwargs)
            record = {"event": "response", "response": response.model_dump(mode="json"),
                      "usage": response.usage.model_dump() if response.usage else None}
        except Exception as exc:
            record = {"event": "error", "error": f"{type(exc).__name__}: {exc}"}
            raise
        finally:
            with self.lock:
                append(self.path, {**record, "call_id": call_id, "role": self.role,
                                   "elapsed_sec": time.monotonic() - started})
        return response

    def close(self):
        self.client.close()
