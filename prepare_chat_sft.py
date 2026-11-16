"""prepare_chat_sft.py — Download high-quality SFT data used for 1B-param models.

Sources:
  1. yahma/alpaca-cleaned  — 52k cleaned Stanford Alpaca instruction pairs
  2. OpenAssistant/oasst1  — 9k high-quality human-annotated conversations

Output:
  data/chat_sft_train.jsonl  — {"instruction": "...", "response": "..."}
  data/chat_sft_val.jsonl

Usage:
    python prepare_chat_sft.py
"""
import json
import random
from pathlib import Path

OUT_TRAIN = Path("data/chat_sft_train.jsonl")
OUT_VAL   = Path("data/chat_sft_val.jsonl")


def load_alpaca_cleaned():
    """Download yahma/alpaca-cleaned — 52k clean instruction/output pairs."""
    pairs = []
    try:
        from datasets import load_dataset
        print("Downloading yahma/alpaca-cleaned...")
        ds = load_dataset("yahma/alpaca-cleaned", trust_remote_code=True)
        for ex in ds["train"]:
            instruction = ex.get("instruction", "").strip()
            inp         = ex.get("input", "").strip()
            output      = ex.get("output", "").strip()
            if not instruction or not output:
                continue
            if len(output) < 20:
                continue
            # Combine instruction + input if input is non-empty
            full_instruction = f"{instruction}\n{inp}" if inp else instruction
            pairs.append({"instruction": full_instruction, "response": output})
        print(f"  alpaca-cleaned: {len(pairs):,} pairs")
    except Exception as e:
        print(f"  alpaca-cleaned skipped: {e}")
    return pairs


def load_oasst1():
    """Download OpenAssistant/oasst1 — extract top-level human→assistant turns."""
    pairs = []
    try:
        from datasets import load_dataset
        print("Downloading OpenAssistant/oasst1...")
        ds = load_dataset("OpenAssistant/oasst1", trust_remote_code=True)

        # Build message tree: id → message
        all_msgs = {}
        for split in ["train", "validation"]:
            if split in ds:
                for msg in ds[split]:
                    all_msgs[msg["message_id"]] = msg

        # Find human→assistant pairs where human has no parent (root turns)
        # and assistant reply is the highest-ranked child
        human_roots = [m for m in all_msgs.values()
                       if m["role"] == "prompter" and m["parent_id"] is None
                       and m["lang"] == "en"]

        for human_msg in human_roots:
            # Find assistant replies to this human message
            replies = [m for m in all_msgs.values()
                       if m["parent_id"] == human_msg["message_id"]
                       and m["role"] == "assistant"]
            if not replies:
                continue
            # Pick highest-ranked reply
            best = max(replies, key=lambda m: m.get("rank", 999) * -1
                       if m.get("rank") is not None else 0)
            instruction = human_msg.get("text", "").strip()
            response    = best.get("text", "").strip()
            if instruction and response and len(response) > 30:
                pairs.append({"instruction": instruction, "response": response})

        print(f"  oasst1: {len(pairs):,} pairs")
    except Exception as e:
        print(f"  oasst1 skipped: {e}")
    return pairs


def load_dolly():
    """Download databricks/databricks-dolly-15k as fallback."""
    pairs = []
    try:
        from datasets import load_dataset
        print("Downloading databricks/databricks-dolly-15k...")
        ds = load_dataset("databricks/databricks-dolly-15k", trust_remote_code=True)
        for ex in ds["train"]:
            instruction = ex.get("instruction", "").strip()
            context     = ex.get("context", "").strip()
            response    = ex.get("response", "").strip()
            if not instruction or not response or len(response) < 20:
                continue
            full_instruction = f"{instruction}\n\nContext: {context}" if context else instruction
            pairs.append({"instruction": full_instruction, "response": response})
        print(f"  dolly-15k: {len(pairs):,} pairs")
    except Exception as e:
        print(f"  dolly-15k skipped: {e}")
    return pairs


def filter_quality(pairs):
    """Remove very short, very long, or clearly broken examples."""
    filtered = []
    for p in pairs:
        instr = p["instruction"]
        resp  = p["response"]
        if len(instr) < 10 or len(resp) < 20:
            continue
        if len(resp) > 4000:
            continue
        # Skip if response is just repetition of instruction
        if resp.strip().lower() == instr.strip().lower():
            continue
        filtered.append(p)
    return filtered


def main():
    random.seed(42)

    all_pairs = []
    all_pairs += load_alpaca_cleaned()
    all_pairs += load_oasst1()
    all_pairs += load_dolly()

    print(f"\nTotal before filtering: {len(all_pairs):,}")
    all_pairs = filter_quality(all_pairs)
    print(f"Total after filtering:  {len(all_pairs):,}")

    random.shuffle(all_pairs)

    # 95/5 train/val split
    cut = int(len(all_pairs) * 0.95)
    train_pairs = all_pairs[:cut]
    val_pairs   = all_pairs[cut:]

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n✓ Chat SFT dataset ready")
    print(f"  Train: {len(train_pairs):,} pairs → {OUT_TRAIN}")
    print(f"  Val:   {len(val_pairs):,} pairs → {OUT_VAL}")
    print(f"\nNext step:")
    print("  python finetune.py --stage sft --config configs/general.yaml \\")
    print("      --base-checkpoint checkpoints/general/step_00052000 \\")
    print("      --data data/chat_sft_train.jsonl --tokenizer tokenizer")


if __name__ == "__main__":
    main()
