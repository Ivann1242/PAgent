# Merged oracle labels (17k + hard remine)

Do **not** overwrite `checkpoints/blind_hint_17k/` — historical IID/HF1 tables use those files.

| File | Meaning |
|--|--|
| `oracle_labels.jsonl` | All flip labels: original 3687 + remine 900 |
| `oracle_labels_dedup.jsonl` | One hint per question: original dedup 1809 + remine 560 |
| `stats.json` | Counts / provenance |

Remine pool: baseline-wrong & previously no-flip under k=6; remine @8k k=12.
