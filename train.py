"""Pre-training entry point.

Usage:
    # Single GPU
    python train.py --config configs/small.yaml

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=4 train.py --config configs/medium.yaml

    # Resume from checkpoint
    python train.py --config configs/small.yaml --resume checkpoints/small/step_00010000
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from model.architecture import Transformer
from model.config import ModelConfig, TrainingConfig
from data.dataset import PretrainingDataset
from data.tokenizer import BPETokenizer, SentencePieceTokenizer
from training.trainer import Trainer
from utils.distributed import setup_distributed, cleanup_distributed, wrap_ddp, is_main_process
from utils.logging import get_logger

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train the transformer LLM")
    parser.add_argument("--config",  type=str, required=True,  help="Path to YAML config file")
    parser.add_argument("--resume",    type=str, default=None,   help="Path to checkpoint to resume from")
    parser.add_argument("--tokenizer", type=str, default=None,   help="Path to trained tokenizer directory")
    parser.add_argument("--max-steps", type=int, default=None,   help="Override config max_steps (for chunked training)")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    model_cfg   = ModelConfig(**raw.get("model", {}))
    train_cfg   = TrainingConfig(**raw.get("training", {}))
    return model_cfg, train_cfg


def main() -> None:
    args = parse_args()
    setup_distributed()

    model_cfg, train_cfg = load_config(args.config)

    # CLI override lets us run N steps at a time without editing the config
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps

    # Tokenizer — auto-detect SentencePiece vs custom BPE
    tokenizer_path = Path(args.tokenizer or "tokenizer")
    if not tokenizer_path.exists():
        logger.warning(
            f"No tokenizer found at '{tokenizer_path}'. "
            "Run: python prepare_data.py"
        )
        sys.exit(1)
    if (tokenizer_path / "type.txt").exists():
        tokenizer = SentencePieceTokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer.load(tokenizer_path)
    logger.info(f"Loaded tokenizer — vocab size: {len(tokenizer)}")

    # Sync vocab size
    model_cfg.vocab_size = len(tokenizer)

    # Datasets
    train_dataset = PretrainingDataset(
        data_dir=train_cfg.data_dir,
        seq_len=model_cfg.max_seq_len,
        split="train",
        shuffle=True,
    )
    val_dataset = PretrainingDataset(
        data_dir=train_cfg.data_dir,
        seq_len=model_cfg.max_seq_len,
        split="val",
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.micro_batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.micro_batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    # Speed flags — free throughput on Ampere/Ada/Blackwell without extra hardware
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmuls (~1.5x faster)
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")      # same effect via PyTorch API
    torch.backends.cudnn.benchmark = True           # auto-tune CUDA kernels for fixed input shapes

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(model_cfg)
    if is_main_process():
        logger.info(f"Model parameters: {model.num_parameters():,}")
        logger.info(f"Non-embedding parameters: {model.num_parameters(non_embedding=True):,}")
    model = model.to(device)   # must be on device BEFORE DDP wrapping

    model = wrap_ddp(model)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_cfg,
        train_config=train_cfg,
        resume_from=args.resume,
    )

    trainer.train()

    cleanup_distributed()


if __name__ == "__main__":
    main()
