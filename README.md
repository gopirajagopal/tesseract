# Claude-Style Transformer LLM — From Scratch

A production-grade, Claude-inspired large language model built entirely from scratch in PyTorch.

## Architecture

Decoder-only transformer with all modern improvements used in Claude / LLaMA-class models:

| Component | Implementation |
|---|---|
| **Normalization** | RMSNorm (pre-norm) |
| **Position encoding** | Rotary Embeddings (RoPE) |
| **Attention** | Grouped Query Attention (GQA) + Flash Attention |
| **FFN activation** | SwiGLU |
| **No bias** | Linear layers without bias |
| **Tied embeddings** | Input/output weight sharing (small/medium) |

## Model Sizes

| Config | Params | Hidden | Layers | Heads (Q/KV) | FFN | Seq Len |
|---|---|---|---|---|---|---|
| `small` | ~125M | 768 | 12 | 12 / 4 | 2048 | 2048 |
| `medium` | ~1B | 2048 | 24 | 16 / 8 | 5504 | 4096 |
| `large` | ~7B | 4096 | 32 | 32 / 8 | 11008 | 4096 |
| `xl` | ~70B | 8192 | 80 | 64 / 8 | 28672 | 8192 |

## Project Structure

```
MODEL/
├── model/
│   ├── config.py           # ModelConfig, TrainingConfig, RLHFConfig
│   ├── normalization.py    # RMSNorm
│   ├── embeddings.py       # TokenEmbedding + RotaryEmbedding (RoPE)
│   ├── attention.py        # GroupedQueryAttention (GQA)
│   ├── feedforward.py      # SwiGLUFFN
│   └── architecture.py     # Transformer (full model)
├── data/
│   ├── tokenizer.py        # BPE tokenizer (train from scratch or wrap sentencepiece)
│   ├── dataset.py          # PretrainingDataset, InstructionDataset, PreferenceDataset
│   └── preprocessing.py    # Text cleaning + binary tokenization pipeline
├── training/
│   ├── trainer.py          # Pre-training loop (mixed precision, grad accumulation)
│   ├── optimizer.py        # AdamW + cosine LR with warmup
│   ├── loss.py             # Cross-entropy, masked cross-entropy
│   └── checkpointing.py    # Save/load/rotate checkpoints
├── rlhf/
│   ├── sft_trainer.py      # Supervised Fine-Tuning (Stage 2)
│   ├── reward_model.py     # Reward model + trainer (Stage 3)
│   ├── dpo_trainer.py      # Direct Preference Optimization
│   ├── ppo_trainer.py      # PPO RLHF (Stage 4)
│   └── constitutional_ai.py# Constitutional AI (Anthropic's CAI pipeline)
├── inference/
│   ├── generate.py         # Generator (KV-cache, streaming, batch, chat)
│   └── sampling.py         # greedy, temperature, top-k, top-p, repetition penalty
├── utils/
│   ├── logging.py          # Logger, MetricsTracker
│   └── distributed.py      # DDP setup, rank helpers
├── configs/
│   ├── small.yaml          # ~125M params
│   ├── medium.yaml         # ~1B params
│   └── large.yaml          # ~7B params
├── train.py                # Pre-training entry point
├── finetune.py             # SFT / DPO / PPO / CAI entry point
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a tokenizer

```python
from data.tokenizer import BPETokenizer

texts = open("data/raw/corpus.txt").readlines()
tok = BPETokenizer(vocab_size=32000)
tok.train(texts)
tok.save("tokenizer/")
```

Or use a pre-trained SentencePiece model:

```python
tok = BPETokenizer.from_sentencepiece("path/to/tokenizer.model")
```

### 3. Preprocess training data

```python
from data.tokenizer import BPETokenizer
from data.preprocessing import TextPreprocessor

tok = BPETokenizer.load("tokenizer/")
proc = TextPreprocessor(tok)
proc.process_directory("data/raw/", "data/processed/", format="txt")
```

### 4. Pre-train

```bash
# Single GPU
python train.py --config configs/small.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train.py --config configs/medium.yaml

# Resume from checkpoint
python train.py --config configs/small.yaml --resume checkpoints/small/step_00010000
```

### 5. Supervised Fine-Tuning (SFT)

Prepare `data/sft.jsonl` — each line:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

```bash
python finetune.py --stage sft \
    --config configs/small.yaml \
    --base-checkpoint checkpoints/small/latest \
    --data data/sft.jsonl
```

### 6. Constitutional AI (CAI)

Generate revised responses using Anthropic's CAI approach:

```bash
python finetune.py --stage cai \
    --config configs/small.yaml \
    --base-checkpoint checkpoints/sft/latest \
    --data data/prompts.jsonl \
    --output data/cai_revised.jsonl
```

### 7. DPO Fine-Tuning (Recommended RLHF)

Prepare `data/preferences.jsonl` — each line:
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

```bash
python finetune.py --stage dpo \
    --config configs/small.yaml \
    --base-checkpoint checkpoints/sft/latest \
    --data data/preferences.jsonl
```

### 8. PPO Fine-Tuning (Full RLHF)

First train a reward model, then run PPO:

```bash
# Train reward model
python finetune.py --stage reward_model \
    --config configs/small.yaml \
    --base-checkpoint checkpoints/sft/latest \
    --data data/preferences.jsonl

# PPO
python finetune.py --stage ppo \
    --config configs/small.yaml \
    --base-checkpoint checkpoints/sft/latest \
    --reward-checkpoint checkpoints/reward_model/latest \
    --data data/prompts.jsonl
```

### 9. Inference / Chat

```python
from model.architecture import Transformer
from model.config import ModelConfig
from data.tokenizer import BPETokenizer
from inference.generate import Generator, chat
from training.checkpointing import Checkpointer

# Load
tok   = BPETokenizer.load("tokenizer/")
cfg   = ModelConfig.small()
model = Transformer(cfg)
Checkpointer("checkpoints/dpo").load(model, checkpoint_path="checkpoints/dpo/latest")

# Single generation
gen  = Generator(model, tok)
text = gen.generate("What is the capital of France?", max_new_tokens=200)
print(text)

# Streaming
for chunk in gen.stream("Explain quantum entanglement simply:"):
    print(chunk, end="", flush=True)

# Interactive chat
chat(model, tok, system_prompt="You are a helpful AI assistant.")
```

## Training Pipeline (Claude-style)

```
Raw Text Corpus
      │
      ▼
1. Pre-training ─────── next-token prediction on trillions of tokens
      │
      ▼
2. SFT ──────────────── supervised fine-tuning on (instruction, response) pairs
      │
      ▼
3. Constitutional AI ── model critiques & revises its own outputs
      │                  generates preference data via AI feedback
      ▼
4a. DPO ─────────────── direct preference optimization (simpler, more stable)
    OR
4b. Reward Model ─────── train on preference data
      │
      ▼
    PPO ──────────────── proximal policy optimization with KL penalty
      │
      ▼
  Aligned Model
```

## Hardware Requirements

| Config | VRAM | Training Time |
|---|---|---|
| `small` (125M) | 8 GB (single GPU) | Days on RTX 4090 |
| `medium` (1B) | 40 GB (2× A100) | Weeks |
| `large` (7B) | 320 GB (4× A100 80GB) | Months |
| `xl` (70B) | 4–8 × H100 80GB | Months + |

## Key Design Decisions

- **RMSNorm** over LayerNorm: faster, no mean subtraction, same or better results
- **RoPE** over learned positions: generalises to longer sequences at inference
- **SwiGLU** over GELU FFN: consistently better perplexity (PaLM, LLaMA, Claude)
- **GQA** over MHA: 2–8× smaller KV cache, faster inference, minimal quality loss
- **DPO** as default RLHF: avoids reward hacking, no separate RM needed, more stable than PPO
- **Constitutional AI**: self-critiquing loop reduces reliance on human preference data
