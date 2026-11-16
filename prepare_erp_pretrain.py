"""prepare_erp_pretrain.py — Build ERP domain pre-training corpus.

Downloads business/finance/ERP text from HuggingFace datasets and tokenizes
into .bin files ready for continued pre-training.

Sources:
  1. reuters21578          — Reuters business news (~21k articles)
  2. financial_phrasebank  — Financial sentences (~5k)
  3. gbharti/finance-alpaca — Finance Q&A as text (~68k examples)
  4. Sujet-Finance-Instruct-177k — Finance instruction text (~177k)

Output: data/erp/ (train_*.bin + val_*.bin)

Usage:
    python prepare_erp_pretrain.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).parent
TOKDIR  = ROOT / "tokenizer"
OUTDIR  = ROOT / "data" / "erp"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tokenizer():
    from data.tokenizer import BPETokenizer, SentencePieceTokenizer
    if (TOKDIR / "type.txt").exists():
        return SentencePieceTokenizer.load(TOKDIR)
    return BPETokenizer.load(TOKDIR)


def write_bin(token_lists: list, out_path: Path) -> int:
    total = 0
    with open(out_path, "wb") as f:
        for ids in token_lists:
            arr = np.array(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            total += len(ids)
    return total


def encode_texts(texts: list, tokenizer, min_chars: int = 80) -> list:
    result = []
    for text in texts:
        text = text.strip()
        if len(text) < min_chars:
            continue
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if ids:
            result.append(ids)
    return result


def split_and_save(all_ids: list, name: str, val_ratio: float = 0.01):
    cut = max(1, int(len(all_ids) * (1 - val_ratio)))
    train_ids = all_ids[:cut]
    val_ids   = all_ids[cut:] or all_ids[:max(1, len(all_ids) // 10)]

    n_train = write_bin(train_ids, OUTDIR / f"train_{name}.bin")
    n_val   = write_bin(val_ids,   OUTDIR / f"val_{name}.bin")
    print(f"  [{name}] train={n_train:,} tokens  val={n_val:,} tokens")
    return n_train


# ---------------------------------------------------------------------------
# Source 1 — Reuters 21578 (business news)
# ---------------------------------------------------------------------------
def prepare_reuters(tokenizer) -> int:
    print("\n[1/4] Reuters 21578 — business news")
    from datasets import load_dataset
    ds = load_dataset("reuters21578", "ModApte", trust_remote_code=True)

    texts = []
    for split in ("train", "test"):
        if split not in ds:
            continue
        for ex in ds[split]:
            title = ex.get("title", "").strip()
            body  = ex.get("text",  "").strip()
            combined = f"{title}. {body}" if title else body
            if combined:
                texts.append(combined)

    print(f"  {len(texts):,} articles")
    ids = encode_texts(texts, tokenizer)
    return split_and_save(ids, "00_reuters")


# ---------------------------------------------------------------------------
# Source 2 — Financial PhraseBank
# ---------------------------------------------------------------------------
def prepare_phrasebank(tokenizer) -> int:
    print("\n[2/4] Financial PhraseBank")
    from datasets import load_dataset
    ds = load_dataset("financial_phrasebank", "sentences_allagree",
                      trust_remote_code=True)

    texts = [ex["sentence"] for split in ds.values() for ex in split
             if ex.get("sentence")]
    print(f"  {len(texts):,} sentences")
    ids = encode_texts(texts, tokenizer, min_chars=20)
    return split_and_save(ids, "01_phrasebank")


# ---------------------------------------------------------------------------
# Source 3 — Finance Alpaca Q&A (use as running text)
# ---------------------------------------------------------------------------
def prepare_finance_alpaca(tokenizer) -> int:
    print("\n[3/4] Finance Alpaca (~68k Q&A)")
    from datasets import load_dataset
    ds = load_dataset("gbharti/finance-alpaca", trust_remote_code=True)

    texts = []
    for ex in ds["train"]:
        instruction = ex.get("instruction", "").strip()
        output      = ex.get("output", "").strip()
        if instruction and output:
            texts.append(f"Question: {instruction}\nAnswer: {output}")

    print(f"  {len(texts):,} examples")
    ids = encode_texts(texts, tokenizer)
    return split_and_save(ids, "02_finance_alpaca")


# ---------------------------------------------------------------------------
# Source 4 — Sujet Finance Instructions (177k)
# ---------------------------------------------------------------------------
def prepare_sujet_finance(tokenizer) -> int:
    print("\n[4/4] Sujet Finance Instruct (177k)")
    from datasets import load_dataset
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k",
                      trust_remote_code=True)

    texts = []
    for ex in ds["train"]:
        # Use instruction + answer as document text
        instr  = ex.get("instruction", ex.get("input",  "")).strip()
        answer = ex.get("output",      ex.get("answer", "")).strip()
        if instr and answer:
            texts.append(f"{instr}\n{answer}")
        elif answer:
            texts.append(answer)

    print(f"  {len(texts):,} examples")
    ids = encode_texts(texts, tokenizer)
    return split_and_save(ids, "03_sujet_finance")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading tokenizer…")
    tokenizer = load_tokenizer()
    print(f"  vocab size: {len(tokenizer):,}")

    total = 0
    try:
        total += prepare_reuters(tokenizer)
    except Exception as e:
        print(f"  Reuters skipped: {e}")

    try:
        total += prepare_phrasebank(tokenizer)
    except Exception as e:
        print(f"  PhraseBank skipped: {e}")

    try:
        total += prepare_finance_alpaca(tokenizer)
    except Exception as e:
        print(f"  Finance Alpaca skipped: {e}")

    try:
        total += prepare_sujet_finance(tokenizer)
    except Exception as e:
        print(f"  Sujet Finance skipped: {e}")

    print(f"\n✓ ERP pre-training corpus ready — {total:,} total tokens")
    print(f"  Files in: {OUTDIR}")
    print("  Next: python train.py --config configs/erp.yaml "
          "--resume checkpoints/general/step_00060000 --tokenizer tokenizer")


if __name__ == "__main__":
    main()
