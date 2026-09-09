"""Freeze a randomly ordered exact-deduplicated JSONL test set."""
import json
from pathlib import Path
import random
import re
import unicodedata

from .io import dataset, file_hash, rows, write_json


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def freeze(candidates, exclusions, output, count, seed):
    output = Path(output)
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise ValueError("Refusing to overwrite a frozen dataset")
    blocked = set()
    sources = {}
    for name in exclusions:
        sources[name] = file_hash(name)
        for row in rows(Path(name)):
            if not isinstance(row.get("problem"), str):
                raise ValueError(f"Exclusion row without problem: {name}")
            blocked.add(normalize(row["problem"]))
    pool = []
    seen = set()
    for row in dataset(candidates):
        key = normalize(row["problem"])
        if key not in blocked and key not in seen:
            seen.add(key)
            pool.append(row)
    if count <= 0 or len(pool) < count:
        raise ValueError(f"Need {count} unique eligible candidates; found {len(pool)}")
    random.Random(seed).shuffle(pool)
    selected = pool[:count]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents replacing a set selected by another process.
    with output.open("x") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output.with_suffix(".manifest.json"), {
        "candidates": str(candidates), "candidates_hash": file_hash(candidates),
        "exclusion_sources": sources, "seed": seed, "n": count,
        "ids": [r["id"] for r in selected], "sha256": file_hash(output),
        "scope": "Exact normalized deduplication only; near-duplicate and development-use audit still required",
    })
