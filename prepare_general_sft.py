"""prepare_general_sft.py — Build a rich general-purpose SFT dataset.

Combines:
  - databricks/databricks-dolly-15k  : 15k human-written instructions across
      8 categories (open QA, closed QA, summarisation, classification,
      brainstorming, creative writing, information extraction, generation)
  - existing data/sft_train.jsonl    : Alpaca 52k (already on disk)

Output (same format as InstructionDataset expects):
  data/general_sft_train.jsonl   — training split
  data/general_sft_val.jsonl     — validation split (~5%)

Usage:
    python prepare_general_sft.py

Then fine-tune (after pre-training finishes):
    python finetune.py --stage sft --config configs/general.yaml \\
        --base-checkpoint checkpoints/general/step_00100000 \\
        --data data/general_sft_train.jsonl --tokenizer tokenizer \\
        --checkpoint-dir checkpoints/general_sft
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

DATA_DIR  = Path("data")
SEED      = 42
VAL_RATIO = 0.05
MAX_DOLLY = 15_000   # use all of Dolly-15k
MAX_ALPACA = 20_000  # cap Alpaca to avoid it dominating


def msg(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "user",      "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]}


def load_dolly() -> list:
    print("Downloading databricks/databricks-dolly-15k …")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    examples = []
    for row in ds:
        instruction = row.get("instruction", "").strip()
        context     = row.get("context", "").strip()
        response    = row.get("response", "").strip()
        if not instruction or not response:
            continue
        user_content = instruction
        if context:
            user_content = f"{instruction}\n\nContext:\n{context}"
        examples.append(msg(user_content, response))
    print(f"  Dolly-15k: {len(examples):,} examples")
    return examples[:MAX_DOLLY]


def load_alpaca_from_disk() -> list:
    path = DATA_DIR / "sft_train.jsonl"
    if not path.exists():
        print("  data/sft_train.jsonl not found — skipping Alpaca")
        return []
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"  Alpaca (on disk): {len(examples):,} examples — using up to {MAX_ALPACA:,}")
    return examples[:MAX_ALPACA]


def main():
    print("=" * 60)
    print("Building general SFT dataset")
    print("=" * 60)

    random.seed(SEED)

    dolly   = load_dolly()
    alpaca  = load_alpaca_from_disk()

    combined = dolly + alpaca
    random.shuffle(combined)

    # Remove blank/short examples
    combined = [
        ex for ex in combined
        if ex["messages"][0]["content"] and ex["messages"][1]["content"]
        and len(ex["messages"][1]["content"]) >= 20
    ]

    n_val    = max(1, int(len(combined) * VAL_RATIO))
    val_ex   = combined[:n_val]
    train_ex = combined[n_val:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "general_sft_train.jsonl"
    val_path   = DATA_DIR / "general_sft_val.jsonl"

    for path, data in [(train_path, train_ex), (val_path, val_ex)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print(f"  Total examples : {len(combined):,}")
    print(f"  Train          : {len(train_ex):,}  → {train_path}")
    print(f"  Val            : {len(val_ex):,}   → {val_path}")
    print("=" * 60)
    print()
    print("Category breakdown (Dolly):")
    from collections import Counter
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    cats = Counter(row["category"] for row in ds)
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {cnt:>5}")
    print()
    print("Next — wait for pre-training to finish, then run:")
    print("  python finetune.py --stage sft --config configs/general.yaml \\")
    print("      --base-checkpoint checkpoints/general/step_00100000 \\")
    print("      --data data/general_sft_train.jsonl --tokenizer tokenizer \\")
    print("      --checkpoint-dir checkpoints/general_sft")


if __name__ == "__main__":
    main()
