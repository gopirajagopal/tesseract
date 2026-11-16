"""prepare_medical_pretrain.py — Download and tokenize medical + broad English data.

Downloads:
  1. PubMed abstracts (ccdv/pubmed-summarization) — medical domain
  2. BookCorpus (bookcorpusopen/books_in_english) — broad English vocabulary

Outputs .bin files into data/general/:
  train_02_medical.bin  / val_02_medical.bin
  train_03_books.bin    / val_03_books.bin

Usage:
    pip install datasets
    python prepare_medical_pretrain.py
    pip install datasets
    python prepare_medical_pretrain.py --medical-only   # skip books
    python prepare_medical_pretrain.py --books-only     # skip medical
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent
TOK_DIR   = ROOT / "tokenizer"
OUT_DIR   = ROOT / "data" / "general"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tokenizer():
    from data.tokenizer import BPETokenizer, SentencePieceTokenizer
    if (TOK_DIR / "type.txt").exists():
        return SentencePieceTokenizer.load(TOK_DIR)
    return BPETokenizer.load(TOK_DIR)


def write_bin(token_lists: list, out_path: Path) -> int:
    """Write a list of token-id lists to a flat uint16 binary file."""
    total = 0
    with open(out_path, "wb") as f:
        for ids in token_lists:
            arr = np.array(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            total += len(ids)
    return total


def tokenize_texts(texts: list, tokenizer, min_len: int = 64) -> list:
    """Tokenize a list of strings, skip blanks/too-short."""
    result = []
    for text in texts:
        text = text.strip()
        if len(text) < min_len:
            continue
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if ids:
            result.append(ids)
    return result


# ---------------------------------------------------------------------------
# Medical — PubMed abstracts
# ---------------------------------------------------------------------------
def prepare_medical(tokenizer):
    print("\n=== Medical: PubMed abstracts (ccdv/pubmed-summarization) ===")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("Downloading PubMed dataset (this may take a while)...")
    ds = load_dataset("ccdv/pubmed-summarization", trust_remote_code=True)

    # Use abstract + article text for richest medical content
    def extract_text(example):
        parts = []
        if example.get("abstract"):
            parts.append(example["abstract"].strip())
        if example.get("article"):
            parts.append(example["article"].strip())
        return " ".join(parts)

    train_texts = [extract_text(ex) for ex in ds["train"]]
    val_texts   = [extract_text(ex) for ex in ds["validation"]]

    print(f"  Train docs: {len(train_texts):,}  |  Val docs: {len(val_texts):,}")
    print("  Tokenizing train split...")
    train_ids = tokenize_texts(train_texts, tokenizer)
    print("  Tokenizing val split...")
    val_ids   = tokenize_texts(val_texts,   tokenizer)

    train_path = OUT_DIR / "train_02_medical.bin"
    val_path   = OUT_DIR / "val_02_medical.bin"
    n_train = write_bin(train_ids, train_path)
    n_val   = write_bin(val_ids,   val_path)
    print(f"  Wrote {n_train:,} train tokens → {train_path}")
    print(f"  Wrote {n_val:,}   val tokens   → {val_path}")
    return n_train + n_val


# ---------------------------------------------------------------------------
# General English — Books corpus
# ---------------------------------------------------------------------------
def prepare_books(tokenizer):
    print("\n=== General English: books_in_english ===")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("Downloading books corpus (streaming, no full download required)...")
    # Stream to avoid downloading 70GB — take first 200k books
    MAX_TRAIN = 200_000
    MAX_VAL   = 2_000

    train_ids, val_ids = [], []
    ds = load_dataset(
        "bookcorpusopen/books_in_english",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    for i, ex in enumerate(ds):
        text = ex.get("text", "").strip()
        if len(text) < 64:
            continue
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if not ids:
            continue
        if i < MAX_TRAIN:
            train_ids.append(ids)
        elif i < MAX_TRAIN + MAX_VAL:
            val_ids.append(ids)
        else:
            break
        if (i + 1) % 10_000 == 0:
            print(f"    processed {i+1:,} docs...")

    train_path = OUT_DIR / "train_03_books.bin"
    val_path   = OUT_DIR / "val_03_books.bin"
    n_train = write_bin(train_ids, train_path)
    n_val   = write_bin(val_ids,   val_path)
    print(f"  Wrote {n_train:,} train tokens → {train_path}")
    print(f"  Wrote {n_val:,}   val tokens   → {val_path}")
    return n_train + n_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medical-only", action="store_true")
    parser.add_argument("--books-only",   action="store_true")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"  Vocab size: {len(tokenizer):,}")

    total = 0
    if not args.books_only:
        total += prepare_medical(tokenizer)
    if not args.medical_only:
        total += prepare_books(tokenizer)

    print(f"\n✓ Total tokens written: {total:,}")
    print("  New .bin files are in data/general/ and will be picked up automatically on next training run.")
    print("  Restart training after this completes to include the new data.")


if __name__ == "__main__":
    main()
