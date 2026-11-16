"""chat.py — Interactive chat inference with the aligned model.

Usage:
    python chat.py --checkpoint checkpoints/dpo/step_00002257 \
                   --config configs/small.yaml \
                   --tokenizer tokenizer
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

from data.tokenizer import BPETokenizer, SentencePieceTokenizer
from inference.generate import Generator
from model.architecture import Transformer
from model.config import ModelConfig, TrainingConfig, RLHFConfig
from training.checkpointing import Checkpointer


SYSTEM_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          Claude-style LLM — Interactive Chat                 ║
║  Type your message and press Enter. Type 'quit' to exit.    ║
╚══════════════════════════════════════════════════════════════╝
"""

CHAT_TEMPLATE = "Human: {user}\nAssistant:"


def build_prompt(history, user_message):
    """Build the full prompt from conversation history + new message."""
    parts = []
    for turn in history:
        parts.append(f"Human: {turn['user']}")
        parts.append(f"Assistant: {turn['assistant']}")
    parts.append(CHAT_TEMPLATE.format(user=user_message))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with aligned LLM")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint dir")
    parser.add_argument("--config", default="configs/small.yaml", help="Model config YAML")
    parser.add_argument("--tokenizer", default="tokenizer", help="Tokenizer directory")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling p")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--no-history", action="store_true", help="Single-turn mode (no context)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    model_cfg = ModelConfig(**raw.get("model", {}))

    # Load tokenizer
    tok_path = Path(args.tokenizer)
    if (tok_path / "type.txt").exists():
        tokenizer = SentencePieceTokenizer.load(tok_path)
    else:
        tokenizer = BPETokenizer.load(tok_path)
    model_cfg.vocab_size = len(tokenizer)
    print(f"Tokenizer loaded — vocab size: {len(tokenizer)}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(model_cfg).to(device)

    ckpt_path = Path(args.checkpoint)
    checkpointer = Checkpointer(str(ckpt_path.parent))
    checkpointer.load(model, optimizer=None, scheduler=None, checkpoint_path=str(ckpt_path))
    model.to(device)
    print(f"Model loaded from {args.checkpoint} ({model_cfg.num_layers} layers, "
          f"{sum(p.numel() for p in model.parameters()):,} params) on {device}")

    # Build generator
    generator = Generator(model, tokenizer, device=device)

    print(SYSTEM_BANNER)
    print(f"  Temperature : {args.temperature}   Top-p : {args.top_p}   "
          f"Top-k : {args.top_k}   Max tokens : {args.max_new_tokens}")
    print(f"  History     : {'disabled' if args.no_history else 'enabled (multi-turn)'}\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "/clear":
            history.clear()
            print("[History cleared]\n")
            continue
        if user_input.lower() == "/help":
            print("Commands: /clear (reset history), quit (exit)\n")
            continue

        prompt = build_prompt([] if args.no_history else history, user_input)

        print("Assistant: ", end="", flush=True)
        response_parts = []
        for chunk in generator.stream(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=1.1,
            stop_tokens=[tokenizer.eos_token_id],
        ):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)

        response = "".join(response_parts).strip()
        print("\n")

        if not args.no_history:
            history.append({"user": user_input, "assistant": response})


if __name__ == "__main__":
    main()
