"""Fine-tuning entry point — SFT, DPO, PPO, and Constitutional AI.

Usage:
    # SFT fine-tuning
    python finetune.py --stage sft --config configs/small.yaml \\
        --base-checkpoint checkpoints/small/latest \\
        --data data/sft_data.jsonl

    # DPO fine-tuning (after SFT)
    python finetune.py --stage dpo --config configs/small.yaml \\
        --base-checkpoint checkpoints/sft/latest \\
        --data data/preference_data.jsonl

    # PPO fine-tuning (after SFT + reward model)
    python finetune.py --stage ppo --config configs/small.yaml \\
        --base-checkpoint checkpoints/sft/latest \\
        --reward-checkpoint checkpoints/reward_model/latest \\
        --data data/prompts.jsonl

    # Constitutional AI data generation
    python finetune.py --stage cai --config configs/small.yaml \\
        --base-checkpoint checkpoints/sft/latest \\
        --data data/harmful_prompts.jsonl \\
        --output data/cai_revised.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from model.architecture import Transformer
from model.config import ModelConfig, TrainingConfig, RLHFConfig
from data.dataset import InstructionDataset, PreferenceDataset
from data.tokenizer import BPETokenizer, SentencePieceTokenizer
from rlhf.sft_trainer import SFTTrainer
from rlhf.reward_model import RewardModel, RewardModelTrainer
from rlhf.dpo_trainer import DPOTrainer
from rlhf.ppo_trainer import PPOTrainer
from rlhf.constitutional_ai import ConstitutionalAI
from training.checkpointing import Checkpointer
from utils.logging import get_logger

logger = get_logger("finetune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the LLM with RLHF")
    parser.add_argument("--stage",   type=str, required=True,
                        choices=["sft", "reward_model", "dpo", "ppo", "cai"],
                        help="Fine-tuning stage")
    parser.add_argument("--config",  type=str, required=True,  help="Path to YAML config")
    parser.add_argument("--base-checkpoint", type=str, required=True,
                        help="Base model checkpoint to fine-tune")
    parser.add_argument("--reward-checkpoint", type=str, default=None,
                        help="Reward model checkpoint (required for PPO)")
    parser.add_argument("--data",    type=str, required=True,  help="Path to training data")
    parser.add_argument("--output",  type=str, default=None,
                        help="Output path for generated data (CAI stage)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer",
                        help="Path to tokenizer directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint output directory")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    model_cfg = ModelConfig(**raw.get("model", {}))
    train_cfg = TrainingConfig(**raw.get("training", {}))
    rlhf_cfg  = RLHFConfig(**raw.get("rlhf", {}))
    return model_cfg, train_cfg, rlhf_cfg


def load_model(checkpoint_path: str, model_cfg: ModelConfig) -> Transformer:
    model = Transformer(model_cfg)
    ckpt  = Checkpointer(str(Path(checkpoint_path).parent))
    ckpt.load(model, checkpoint_path=checkpoint_path)
    return model


def main() -> None:
    args = parse_args()
    model_cfg, train_cfg, rlhf_cfg = load_config(args.config)

    # Tokenizer — auto-detect SentencePiece vs custom BPE
    tok_path = Path(args.tokenizer)
    if (tok_path / "type.txt").exists():
        tokenizer = SentencePieceTokenizer.load(tok_path)
    else:
        tokenizer = BPETokenizer.load(tok_path)
    model_cfg.vocab_size = len(tokenizer)
    logger.info(f"Tokenizer loaded — vocab size: {len(tokenizer)}")

    # Base model
    logger.info(f"Loading base model from {args.base_checkpoint}")
    model = load_model(args.base_checkpoint, model_cfg)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # ---------------------------------------------------------------
    if args.stage == "sft":
        logger.info("Stage: Supervised Fine-Tuning (SFT)")
        dataset = InstructionDataset(args.data, tokenizer, max_length=model_cfg.max_seq_len)
        n_val   = max(1, int(len(dataset) * 0.05))
        train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
        train_loader = DataLoader(train_ds, batch_size=rlhf_cfg.sft_batch_size, shuffle=True,  num_workers=train_cfg.num_workers)
        val_loader   = DataLoader(val_ds,   batch_size=rlhf_cfg.sft_batch_size, shuffle=False, num_workers=train_cfg.num_workers)

        ckpt_dir = args.checkpoint_dir or "checkpoints/sft"
        trainer = SFTTrainer(model, train_loader, val_loader, rlhf_cfg,
                             checkpoint_dir=ckpt_dir)
        trainer.train()

    # ---------------------------------------------------------------
    elif args.stage == "reward_model":
        logger.info("Stage: Reward Model Training")
        dataset = PreferenceDataset(args.data, tokenizer, max_length=model_cfg.max_seq_len)
        n_val   = max(1, int(len(dataset) * 0.05))
        train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
        train_loader = DataLoader(train_ds, batch_size=rlhf_cfg.rm_batch_size, shuffle=True,  num_workers=train_cfg.num_workers)
        val_loader   = DataLoader(val_ds,   batch_size=rlhf_cfg.rm_batch_size, shuffle=False, num_workers=train_cfg.num_workers)

        reward_model = RewardModel(model, model_cfg)
        ckpt_dir = args.checkpoint_dir or "checkpoints/reward_model"
        trainer = RewardModelTrainer(reward_model, train_loader, val_loader, rlhf_cfg,
                                     checkpoint_dir=ckpt_dir)
        trainer.train()

    # ---------------------------------------------------------------
    elif args.stage == "dpo":
        logger.info("Stage: Direct Preference Optimization (DPO)")
        dataset = PreferenceDataset(args.data, tokenizer, max_length=512)
        n_val   = max(1, int(len(dataset) * 0.05))
        train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
        train_loader = DataLoader(train_ds, batch_size=rlhf_cfg.rm_batch_size, shuffle=True,  num_workers=train_cfg.num_workers)
        val_loader   = DataLoader(val_ds,   batch_size=rlhf_cfg.rm_batch_size, shuffle=False, num_workers=train_cfg.num_workers)

        trainer = DPOTrainer(model, train_loader, val_loader, rlhf_cfg,
                             checkpoint_dir="checkpoints/dpo")
        trainer.train()

    # ---------------------------------------------------------------
    elif args.stage == "ppo":
        if not args.reward_checkpoint:
            logger.error("--reward-checkpoint is required for PPO stage")
            sys.exit(1)

        logger.info("Stage: PPO RLHF")
        reward_base = load_model(args.reward_checkpoint, model_cfg)
        reward_model = RewardModel(reward_base, model_cfg)

        # Load prompts
        prompts = []
        with open(args.data, "r") as f:
            for line in f:
                obj = json.loads(line.strip())
                prompts.append(obj.get("prompt", obj.get("text", "")))

        ppo_trainer = PPOTrainer(
            policy_model=model,
            reward_model=reward_model,
            hidden_size=model_cfg.hidden_size,
            config=rlhf_cfg,
            checkpoint_dir="checkpoints/ppo",
        )

        batch_size = rlhf_cfg.ppo_batch_size
        for step, i in enumerate(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i : i + batch_size]
            encoded = tokenizer.batch_encode(
                batch_prompts, max_length=512, padding=True, truncation=True,
                add_bos=True, add_eos=False,
            )
            input_ids      = torch.tensor(encoded["input_ids"])
            attention_mask = torch.tensor(encoded["attention_mask"])

            metrics = ppo_trainer.train_step(input_ids, attention_mask, step=step)
            if step % 10 == 0:
                logger.info(f"PPO step {step} | {metrics}")
            if step % 100 == 0:
                ppo_trainer.save(step=step, loss=metrics.get("loss", 0.0))

    # ---------------------------------------------------------------
    elif args.stage == "cai":
        logger.info("Stage: Constitutional AI data generation")
        prompts = []
        with open(args.data, "r") as f:
            for line in f:
                obj = json.loads(line.strip())
                prompts.append(obj.get("prompt", obj.get("text", "")))

        cai = ConstitutionalAI(model, tokenizer, rlhf_cfg)
        dataset = cai.process_batch(prompts, verbose=True)

        output_path = args.output or "data/cai_output.jsonl"
        cai.save_dataset(dataset, output_path)
        logger.info(f"CAI dataset saved to {output_path}")

        # Also build AI-feedback preference data for reward model training
        pref_path = output_path.replace(".jsonl", "_preferences.jsonl")
        pref_data = cai.build_preference_dataset(prompts[:min(len(prompts), 500)])
        cai.save_dataset(pref_data, pref_path)
        logger.info(f"AI feedback preference dataset saved to {pref_path}")


if __name__ == "__main__":
    main()
