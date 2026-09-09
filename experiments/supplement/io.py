from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_hash(path: str | Path) -> str:
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file()
                   and p.suffix in {".json", ".safetensors", ".bin", ".model", ".jinja"}
                   and "snapshots" not in p.relative_to(path).parts)
    if not files:
        raise ValueError(f"No model files: {path}")
    return digest({str(p.relative_to(path)): file_hash(p) for p in files})


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dataset(path: str) -> list[dict]:
    result = rows(Path(path))
    ids = [str(r["id"]) for r in result]
    if not result or len(set(ids)) != len(ids):
        raise ValueError("Dataset must be nonempty with unique IDs")
    for row in result:
        if not isinstance(row.get("problem"), str) or not row["problem"].strip():
            raise ValueError("Every row requires a nonempty problem")
        if row.get("gold") is None or str(row["gold"]).strip() == "":
            raise ValueError("Every row requires gold for offline scoring")
    return result
