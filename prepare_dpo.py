"""prepare_dpo.py — Download Anthropic HH-RLHF and convert to PreferenceDataset format.

Each output line:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    python prepare_dpo.py
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

DATA_DIR  = Path("data")
VAL_RATIO = 0.05
SEED      = 42
MAX_TRAIN = 10_000   # cap to keep training manageable


def parse_hh(text: str):
    """Split HH-RLHF text into (prompt, response) by last Assistant turn."""
    parts = re.split(r"\n\nAssistant:", text)
    if len(parts) < 2:
        return None, None
    response = parts[-1].strip()
    prompt   = "\n\nAssistant:".join(parts[:-1]).strip()
    return prompt, response


def main() -> None:
    print("=" * 60)
    print("Downloading Anthropic/hh-rlhf …")
    print("=" * 60)

    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"  Downloaded {len(ds):,} examples")

    examples = []
    skipped  = 0
    for ex in ds:
        prompt_c, chosen   = parse_hh(ex["chosen"])
        prompt_r, rejected = parse_hh(ex["rejected"])

        if not chosen or not rejected or not prompt_c:
            skipped += 1
            continue

        examples.append({
            "prompt":   prompt_c,
            "chosen":   chosen,
            "rejected": rejected,
        })

    print(f"  Converted : {len(examples):,}")
    print(f"  Skipped   : {skipped}")

    random.seed(SEED)
    random.shuffle(examples)
    examples = examples[:MAX_TRAIN]

    n_val    = max(1, int(len(examples) * VAL_RATIO))
    val_ex   = examples[:n_val]
    train_ex = examples[n_val:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "dpo_train.jsonl"
    val_path   = DATA_DIR / "dpo_val.jsonl"

    for path, data in [(train_path, train_ex), (val_path, val_ex)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Train : {len(train_ex):,}  → {train_path}")
    print(f"  Val   : {len(val_ex):,}   → {val_path}")
    print("\nNext step:")
    print("  python finetune.py --stage dpo --config configs/small.yaml \\")
    print("      --base-checkpoint checkpoints/sft/step_00017592 \\")
    print("      --data data/dpo_train.jsonl --tokenizer tokenizer")


if __name__ == "__main__":
    main()
