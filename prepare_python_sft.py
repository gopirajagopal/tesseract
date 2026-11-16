"""prepare_python_sft.py — Download Python coding instruction data for domain SFT.

Downloads iamtarun/python_code_instructions_18k_alpaca (~18K examples) and writes:
    data/python_sft_train.jsonl  — training split
    data/python_sft_val.jsonl    — validation split (~5%)

Each line:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python prepare_python_sft.py
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

DATA_DIR  = Path("data")
VAL_RATIO = 0.05
SEED      = 42
MAX_TRAIN = 15_000   # cap so fine-tuning stays manageable


def to_messages(example: dict) -> dict | None:
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
    print("Downloading iamtarun/python_code_instructions_18k_alpaca …")
    print("=" * 60)

    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    print(f"  Downloaded {len(ds):,} examples")

    examples, skipped = [], 0
    for ex in ds:
        converted = to_messages(ex)
        if converted is None:
            skipped += 1
        else:
            examples.append(converted)

    print(f"  Converted : {len(examples):,}  |  Skipped: {skipped}")

    random.seed(SEED)
    random.shuffle(examples)
    examples = examples[:MAX_TRAIN]

    n_val    = max(1, int(len(examples) * VAL_RATIO))
    val_ex   = examples[:n_val]
    train_ex = examples[n_val:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "python_sft_train.jsonl"
    val_path   = DATA_DIR / "python_sft_val.jsonl"

    for path, data in [(train_path, train_ex), (val_path, val_ex)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Train : {len(train_ex):,}  → {train_path}")
    print(f"  Val   : {len(val_ex):,}   → {val_path}")
    print("\nNext step:")
    print("  python finetune.py --stage sft --config configs/small.yaml \\")
    print("      --base-checkpoint checkpoints/dpo/step_00002257 \\")
    print("      --data data/python_sft_train.jsonl --tokenizer tokenizer \\")
    print("      --checkpoint-dir checkpoints/python_sft")


if __name__ == "__main__":
    main()
