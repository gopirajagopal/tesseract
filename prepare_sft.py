"""prepare_sft.py — Download Alpaca instruction data and convert to InstructionDataset format.

Downloads tatsu-lab/alpaca (~52K examples) from HuggingFace and writes:
    data/sft_train.jsonl  — training split
    data/sft_val.jsonl    — validation split (~5%)

Each line is:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python prepare_sft.py
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

DATA_DIR  = Path("data")
VAL_RATIO = 0.05
SEED      = 42


def alpaca_to_messages(example: dict) -> dict | None:
    """Convert one Alpaca example to messages format."""
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    if not instruction or not output:
        return None

    user_content = instruction
    if inp:
        user_content = f"{instruction}\n\n{inp}"

    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def main() -> None:
    print("=" * 60)
    print("Downloading tatsu-lab/alpaca …")
    print("=" * 60)

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  Downloaded {len(ds):,} examples")

    examples = []
    skipped  = 0
    for ex in ds:
        converted = alpaca_to_messages(ex)
        if converted is None:
            skipped += 1
        else:
            examples.append(converted)

    print(f"  Converted  : {len(examples):,}")
    print(f"  Skipped    : {skipped}")

    random.seed(SEED)
    random.shuffle(examples)

    n_val   = max(1, int(len(examples) * VAL_RATIO))
    val_ex  = examples[:n_val]
    train_ex = examples[n_val:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "sft_train.jsonl"
    val_path   = DATA_DIR / "sft_val.jsonl"

    for path, data in [(train_path, train_ex), (val_path, val_ex)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Train : {len(train_ex):,}  → {train_path}")
    print(f"  Val   : {len(val_ex):,}   → {val_path}")
    print("\nNext step:")
    print("  python finetune.py --stage sft --config configs/small.yaml \\")
    print("      --base-checkpoint checkpoints/small/step_00010000 \\")
    print("      --data data/sft_train.jsonl --tokenizer tokenizer")


if __name__ == "__main__":
    main()
