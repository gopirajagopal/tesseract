"""demo_train.py — End-to-end CPU training demo.

Trains a nano transformer on synthetic text to verify the full stack.
No GPU required.  Runtime: ~3-6 minutes on a modern CPU.

Usage:
    python demo_train.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).parent))

from model.config import ModelConfig, TrainingConfig
from model.architecture import Transformer
from data.tokenizer import BPETokenizer
from training.trainer import Trainer

# ── 1. Synthetic corpus ────────────────────────────────────────────────
SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "a language model learns to predict the next token given its context",
    "transformers use self attention to capture long range dependencies",
    "the cat sat on the mat and looked at the rat",
    "deep learning models are trained on large amounts of text data",
    "the sky is blue and the grass is green today",
    "neural networks learn representations through gradient descent optimization",
    "language models can generate coherent and fluent text when trained well",
    "attention is all you need for sequence to sequence learning tasks",
    "the model uses rotary position embeddings for positional encoding",
    "layer normalization is applied before each transformer decoder block",
    "the feed forward network uses a swish gated linear unit activation",
    "grouped query attention reduces the memory footprint during inference time",
    "the model is trained with the adam optimizer and cosine annealing",
    "constitutional ai provides alignment through self critique and revision",
    "machine learning is a subset of artificial intelligence research",
    "the transformer architecture was introduced in attention is all you need",
    "large language models are pre trained on diverse internet text corpora",
    "reinforcement learning from human feedback improves model alignment",
    "direct preference optimization is a simpler alternative to ppo training",
    "the reward model learns to predict human preferences from pairwise data",
    "supervised fine tuning adapts the base model to follow instructions",
    "the model generates text by sampling from the predicted token distribution",
    "temperature controls the randomness of the sampling process during inference",
    "top k sampling restricts token selection to the k most likely candidates",
    "nucleus sampling selects from the smallest set of tokens above a threshold",
    "the key value cache speeds up autoregressive decoding significantly",
    "gradient checkpointing trades compute for memory during model training",
    "the tokenizer converts raw text into integer token ids for the model",
    "byte pair encoding is a subword tokenization algorithm used in modern nlp",
    "the vocabulary size determines how many unique tokens the model knows",
    "pre training on unlabeled data followed by fine tuning is the standard approach",
    "the loss function measures how well the model predicts the next token",
    "backpropagation computes gradients of the loss with respect to model parameters",
    "weight decay regularizes the model by penalizing large parameter values",
    "the learning rate schedule controls how the step size changes during training",
    "dropout randomly zeroes activations during training to prevent overfitting",
    "residual connections allow gradients to flow directly through the network",
    "the softmax function converts raw logits into a probability distribution",
    "embeddings map discrete tokens to continuous vector representations",
]

CORPUS = SENTENCES * 80   # 3200 sentences → plenty of tokens

# ── 2. Train BPE tokenizer ─────────────────────────────────────────────
print("=" * 60)
print("Step 1/5  Training BPE tokenizer (vocab_size=512)")
print("=" * 60)
tokenizer = BPETokenizer(vocab_size=512)
tokenizer.train(CORPUS, verbose=True)
VOCAB_SIZE = len(tokenizer)
print(f"Final vocab size: {VOCAB_SIZE}\n")

# ── 3. Tokenise & build TensorDataset ──────────────────────────────────
print("Step 2/5  Tokenising corpus ...")
SEQ_LEN = 128
all_ids: list = []
for text in CORPUS:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    all_ids.extend(ids)

total_tokens = len(all_ids)
print(f"Total tokens: {total_tokens:,}")

tokens = torch.tensor(all_ids, dtype=torch.long)
n_seqs = (len(tokens) - 1) // SEQ_LEN
if n_seqs < 4:
    print("ERROR: corpus too small — add more sentences.")
    sys.exit(1)

inputs  = torch.stack([tokens[i * SEQ_LEN     : i * SEQ_LEN + SEQ_LEN    ] for i in range(n_seqs)])
targets = torch.stack([tokens[i * SEQ_LEN + 1 : i * SEQ_LEN + SEQ_LEN + 1] for i in range(n_seqs)])

val_size   = max(2, n_seqs // 10)
train_size = n_seqs - val_size
train_ds, val_ds = random_split(TensorDataset(inputs, targets), [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)
print(f"Sequences  — train: {len(train_ds)}  val: {len(val_ds)}\n")

# ── 4. Build model ─────────────────────────────────────────────────────
print("Step 3/5  Building nano transformer ...")
model_cfg = ModelConfig(
    vocab_size=VOCAB_SIZE,
    max_seq_len=SEQ_LEN,
    hidden_size=128,
    num_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=512,
    use_flash_attention=False,
    use_gradient_checkpointing=False,
)
train_cfg = TrainingConfig(
    learning_rate=5e-4,
    min_learning_rate=5e-5,
    warmup_steps=20,
    max_steps=300,
    micro_batch_size=4,
    gradient_accumulation_steps=1,
    dtype="float32",
    eval_every_steps=50,
    save_every_steps=300,
    log_every_steps=10,
    checkpoint_dir="checkpoints/nano",
    keep_last_n_checkpoints=1,
    num_workers=0,
)

model = Transformer(model_cfg)
print(f"Parameters             : {model.num_parameters():,}")
print(f"Non-embedding params   : {model.num_parameters(non_embedding=True):,}\n")

# ── 5. Pre-train ───────────────────────────────────────────────────────
print("Step 4/5  Pre-training (300 steps, CPU) ...")
print("=" * 60)
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    model_config=model_cfg,
    train_config=train_cfg,
)
trainer.train()

# ── 6. Save tokenizer alongside checkpoint ─────────────────────────────
Path("checkpoints/nano").mkdir(parents=True, exist_ok=True)
tokenizer.save("checkpoints/nano/tokenizer")
print("Tokenizer saved → checkpoints/nano/tokenizer/\n")

# ── 7. Inference test ──────────────────────────────────────────────────
print("Step 5/5  Inference test ...")
print("=" * 60)
model.eval()
from inference.generate import Generator

gen = Generator(model, tokenizer, device="cpu")
prompts = [
    "the quick brown fox",
    "neural networks learn",
    "the model generates text",
    "attention is all you",
]
for prompt in prompts:
    out = gen.generate(prompt, max_new_tokens=15, temperature=0.8, top_k=10)
    print(f"  Prompt : {prompt!r}")
    print(f"  Output : {out!r}")
    print()

print("=" * 60)
print("Demo training complete!  Checkpoint: checkpoints/nano/")
print("=" * 60)
