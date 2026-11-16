"""prepare_data.py — Download TinyStories, train SentencePiece tokenizer, write .bin files.

Usage:
    python prepare_data.py

Outputs:
    tokenizer/spm.model        — SentencePiece BPE model
    tokenizer/spm.vocab        — human-readable vocabulary
    tokenizer/type.txt         — marker so load() knows which class to use
    data/processed/train.bin   — uint16 token array for training
    data/processed/val.bin     — uint16 token array for validation
"""
import sys
import struct
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.tokenizer import SentencePieceTokenizer

# ── Config ─────────────────────────────────────────────────────────────
VOCAB_SIZE        = 8192          # matches configs/small.yaml
TOKENIZER_DIR     = "tokenizer"
DATA_DIR          = Path("data/processed")
TOKENIZER_TRAIN_N = 200_000       # stories used for tokenizer training
DATASET_NAME      = "roneneldan/TinyStories"


def main() -> None:
    from datasets import load_dataset

    # ── 1. Download dataset ────────────────────────────────────────────
    print("=" * 60)
    print("Step 1/4  Downloading TinyStories …")
    print("=" * 60)
    train_ds = load_dataset(DATASET_NAME, split="train",      trust_remote_code=True)
    val_ds   = load_dataset(DATASET_NAME, split="validation", trust_remote_code=True)
    print(f"  train: {len(train_ds):,} stories")
    print(f"  val  : {len(val_ds):,} stories")

    # ── 2. Train SentencePiece tokenizer ───────────────────────────────
    print(f"\nStep 2/4  Training SentencePiece BPE (vocab={VOCAB_SIZE}) …")
    print("=" * 60)
    n = min(TOKENIZER_TRAIN_N, len(train_ds))
    texts = [train_ds[i]["text"] for i in range(n)]

    tok = SentencePieceTokenizer(vocab_size=VOCAB_SIZE)
    tok.train(texts, model_prefix=f"{TOKENIZER_DIR}/spm")
    tok.save(TOKENIZER_DIR)
    print(f"  Tokenizer saved → {TOKENIZER_DIR}/")

    # ── 3. Tokenise & write .bin files ─────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, ds in [("train", train_ds), ("val", val_ds)]:
        print(f"\nStep 3/4  Tokenising {split_name} ({len(ds):,} stories) …")
        out_path = DATA_DIR / f"{split_name}.bin"

        # Stream tokens into a binary file directly to avoid OOM
        total_tokens = 0
        with open(out_path, "wb") as f:
            for item in tqdm(ds, desc=f"  {split_name}", unit="story"):
                ids = tok.encode(item["text"], add_bos=True, add_eos=True)
                arr = np.array(ids, dtype=np.uint16)
                f.write(arr.tobytes())
                total_tokens += len(ids)

        size_mb = out_path.stat().st_size / 1e6
        print(f"  → {total_tokens:,} tokens  ({size_mb:.1f} MB)  {out_path}")

    # ── 4. Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4/4  Data preparation complete!")
    print("=" * 60)
    print(f"  Tokenizer : {TOKENIZER_DIR}/")
    print(f"  Train bin : {DATA_DIR}/train.bin")
    print(f"  Val bin   : {DATA_DIR}/val.bin")
    print()
    print("Next step:")
    print("  python train.py --config configs/small.yaml --tokenizer tokenizer")


if __name__ == "__main__":
    main()
