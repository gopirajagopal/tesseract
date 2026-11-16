import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import ModelConfig, TrainingConfig
from training.loss import cross_entropy_loss
from training.optimizer import build_optimizer, build_lr_scheduler
from training.checkpointing import Checkpointer


class Trainer:
    """Pre-training loop for the transformer language model.

    Features:
    - Mixed precision (bfloat16 / float16)
    - Gradient accumulation
    - Gradient clipping
    - Cosine LR schedule with warmup
    - Periodic evaluation
    - Checkpoint saving / resuming
    - Optional W&B logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        model_config: ModelConfig,
        train_config: TrainingConfig,
        resume_from: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.cfg = train_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model is already moved to device and optionally DDP-wrapped by the caller
        if not hasattr(model, "module"):
            self.model.to(self.device)

        # Mixed precision
        self.dtype = {
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }[train_config.dtype]
        self.autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=(self.dtype == torch.float16))
            if self.device.type == "cuda"
            else None
        )

        self.optimizer = build_optimizer(
            model,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            beta1=train_config.beta1,
            beta2=train_config.beta2,
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
            min_lr_ratio=train_config.min_learning_rate / train_config.learning_rate,
        )
        self.checkpointer = Checkpointer(
            train_config.checkpoint_dir,
            keep_last_n=train_config.keep_last_n_checkpoints,
        )

        self.step = 0
        self.best_val_loss = float("inf")

        if resume_from:
            meta = self.checkpointer.load(
                self.model, self.optimizer, self.scheduler,
                checkpoint_path=resume_from,
            )
            self.step = meta.get("step", 0)

        self._init_wandb()

    def _init_wandb(self) -> None:
        if self.cfg.wandb_project:
            try:
                import wandb
                wandb.init(project=self.cfg.wandb_project, config={
                    **vars(self.model_config),
                    **vars(self.cfg),
                })
                self._wandb = wandb
            except ImportError:
                print("wandb not installed — skipping W&B logging.")
                self._wandb = None
        else:
            self._wandb = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        print(f"Starting training on {self.device}")
        bare_model = self.model.module if hasattr(self.model, "module") else self.model
        print(f"Model parameters: {bare_model.num_parameters():,}")
        print(f"Steps: {self.step} → {self.cfg.max_steps}")

        self.model.train()
        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        t0 = time.time()

        while self.step < self.cfg.max_steps:
            # Gradient accumulation loop
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(self.cfg.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids, targets = batch
                input_ids = input_ids.to(self.device, non_blocking=True)
                targets   = targets.to(self.device, non_blocking=True)

                with self.autocast_ctx:
                    outputs = self.model(input_ids)
                    loss = cross_entropy_loss(
                        outputs["logits"],
                        targets,
                        ignore_index=self.model_config.pad_token_id,
                    )
                    loss = loss / self.cfg.gradient_accumulation_steps

                accum_loss += loss.item()
                self._backward(loss)

            # Gradient clipping + optimizer step
            self._clip_and_step()
            self.scheduler.step()
            self.step += 1

            # Logging
            if self.step % self.cfg.log_every_steps == 0:
                elapsed = time.time() - t0
                lr = self.optimizer.param_groups[0]["lr"]
                tokens_per_sec = (
                    self.cfg.log_every_steps
                    * self.cfg.micro_batch_size
                    * self.model_config.max_seq_len
                    / elapsed
                )
                print(
                    f"step={self.step:6d} | loss={accum_loss:.4f} | "
                    f"lr={lr:.2e} | tokens/s={tokens_per_sec:,.0f}"
                )
                if self._wandb:
                    self._wandb.log({
                        "train/loss": accum_loss,
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                    }, step=self.step)
                t0 = time.time()

            # Evaluation
            if self.val_loader is not None and self.step % self.cfg.eval_every_steps == 0:
                val_loss = self.evaluate()
                print(f"  val_loss={val_loss:.4f}")
                if self._wandb:
                    self._wandb.log({"val/loss": val_loss}, step=self.step)
                self.model.train()

            # Checkpointing
            if self.step % self.cfg.save_every_steps == 0:
                self.checkpointer.save(
                    self.model, self.optimizer, self.scheduler,
                    step=self.step, loss=accum_loss,
                )

        print("Training complete.")
        self.checkpointer.save(
            self.model, self.optimizer, self.scheduler,
            step=self.step, loss=accum_loss, extra={"final": True},
        )

    def _backward(self, loss: torch.Tensor) -> None:
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _clip_and_step(self) -> None:
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, num_batches: int = 50) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0

        for i, batch in enumerate(self.val_loader):
            if i >= num_batches:
                break
            input_ids, targets = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            targets   = targets.to(self.device, non_blocking=True)

            with self.autocast_ctx:
                outputs = self.model(input_ids)
                loss = cross_entropy_loss(
                    outputs["logits"],
                    targets,
                    ignore_index=self.model_config.pad_token_id,
                )
            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)
