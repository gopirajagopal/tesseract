"""prepare_general_pretrain.py — Download and tokenize general English corpora.

Downloads:
  - WikiText-103   : Clean Wikipedia text (~103M tokens) → factual knowledge
  - TinyStories    : Short English stories (~476M tokens) → coherent language

Writes tokenized .bin files to data/general/ using the existing
TextPreprocessor pipeline (uint16 token IDs, streamed).

Usage:
    python prepare_general_pretrain.py

Then train:
    python train.py --config configs/general.yaml \\
                    --resume checkpoints/small/step_00010000
"""
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset

from data.tokenizer import BPETokenizer, SentencePieceTokenizer
from data.preprocessing import TextPreprocessor

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR    = Path("data/general")
TOK_PATH   = Path("tokenizer")

# Cap per dataset so the first run finishes in reasonable time.
# Set to None to use the full dataset.
MAX_WIKI   = 80_000    # WikiText-103 articles  (~80M tokens)
MAX_STORIES = 500_000  # TinyStories stories    (~200M tokens)

SEED = 42
VAL_RATIO = 0.005       # 0.5% validation split


# ── Helpers ───────────────────────────────────────────────────────────────────
def write_bin(token_ids_iter, path: Path) -> int:
    """Write an iterable of token-id lists to a flat uint16 .bin file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(path, "wb") as f:
        for ids in token_ids_iter:
            arr = np.array(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            total += len(ids)
    return total


def process_texts(texts, preprocessor: TextPreprocessor):
    """Clean, filter, and tokenize a list of strings.
    Yields token-id lists for accepted documents."""
    kept = skipped = 0
    for text in texts:
        cleaned = preprocessor.clean(text)
        if cleaned is None:
            skipped += 1
            continue
        ids = preprocessor.tokenizer.encode(cleaned, add_bos=True, add_eos=True)
        kept += 1
        yield ids
    print(f"  Kept {kept:,}  |  Skipped {skipped:,}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    if (TOK_PATH / "type.txt").exists():
        tokenizer = SentencePieceTokenizer.load(TOK_PATH)
    else:
        tokenizer = BPETokenizer.load(TOK_PATH)
    print(f"Tokenizer loaded  — vocab size: {len(tokenizer)}")

    preprocessor = TextPreprocessor(tokenizer, min_length=64, max_length=100_000, dedup=True)

    # ── 1. WikiText-103 ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Dataset 1: WikiText-103  (Wikipedia articles)")
    print("=" * 60)

    wiki_train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                               split="train",  streaming=False)
    wiki_val   = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                               split="validation", streaming=False)

    def iter_wiki(ds, limit=None):
        buf, count = [], 0
        for row in ds:
            text = row["text"].strip()
            if not text or text.startswith(" ="):   # skip headings
                if buf:
                    yield "\n".join(buf)
                    buf = []
                continue
            buf.append(text)
            # Flush every ~10 lines as one document
            if len(buf) >= 10:
                yield "\n".join(buf)
                buf = []
                count += 1
                if limit and count >= limit:
                    break
        if buf:
            yield "\n".join(buf)

    print("  Processing train split…")
    wiki_train_ids = list(process_texts(iter_wiki(wiki_train, MAX_WIKI), preprocessor))
    print("  Processing val split…")
    wiki_val_ids   = list(process_texts(iter_wiki(wiki_val), preprocessor))

    n = write_bin(iter(wiki_train_ids), OUT_DIR / "train_00_wiki.bin")
    print(f"  Wrote train_00_wiki.bin  — {n:,} tokens")
    n = write_bin(iter(wiki_val_ids),   OUT_DIR / "val_00_wiki.bin")
    print(f"  Wrote val_00_wiki.bin    — {n:,} tokens")

    # ── 2. TinyStories ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Dataset 2: TinyStories  (short English stories)")
    print("=" * 60)

    stories_train = load_dataset("roneneldan/TinyStories", split="train",  streaming=False)
    stories_val   = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)

    def iter_stories(ds, limit=None):
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            yield row.get("text", row.get("story", ""))

    print("  Processing train split…")
    story_train_ids = list(process_texts(iter_stories(stories_train, MAX_STORIES), preprocessor))
    print("  Processing val split…")
    story_val_ids   = list(process_texts(iter_stories(stories_val), preprocessor))

    n = write_bin(iter(story_train_ids), OUT_DIR / "train_01_stories.bin")
    print(f"  Wrote train_01_stories.bin — {n:,} tokens")
    n = write_bin(iter(story_val_ids),   OUT_DIR / "val_01_stories.bin")
    print(f"  Wrote val_01_stories.bin   — {n:,} tokens")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(
        os.path.getsize(p) // 2
        for p in OUT_DIR.glob("*.bin")
    )
    print("\n" + "=" * 60)
    print(f"  Output dir  : {OUT_DIR.resolve()}")
    print(f"  Total tokens: ~{total:,}  ({total/1e6:.1f}M)")
    print("=" * 60)
    print("\nNext step — continue pre-training:")
    print("  python train.py --config configs/general.yaml \\")
    print("      --resume checkpoints/small/step_00010000")


if __name__ == "__main__":
    main()
