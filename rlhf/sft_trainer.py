import signal
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import RLHFConfig
from training.loss import masked_cross_entropy_loss
from training.optimizer import build_optimizer, build_lr_scheduler
from training.checkpointing import Checkpointer


class SFTTrainer:
    """Supervised Fine-Tuning (SFT) trainer.

    Fine-tunes the pre-trained base model on instruction/response pairs.
    Loss is computed only on assistant response tokens (loss_mask).
    This is Stage 2 of the RLHF pipeline (Base → SFT model).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: RLHFConfig,
        checkpoint_dir: str = "checkpoints/sft",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.checkpointer = Checkpointer(checkpoint_dir, keep_last_n=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = build_optimizer(
            model,
            learning_rate=config.sft_learning_rate,
            weight_decay=0.01,
        )
        total_steps = config.sft_epochs * (
            len(train_loader) if hasattr(train_loader, "__len__") else 1000
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            warmup_steps=min(50, max(5, total_steps // 10)),
            max_steps=total_steps,
        )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def train(self, save_every_steps: int = 5000) -> None:
        print(f"Starting SFT training for {self.cfg.sft_epochs} epoch(s)")
        global_step = 0
        self._stop_requested = False

        # Graceful stop on Ctrl-C / SIGTERM — saves checkpoint then exits
        def _handle_signal(signum, frame):
            print(f"\n[SFT] Signal {signum} received — saving checkpoint and stopping...")
            self._stop_requested = True

        signal.signal(signal.SIGINT,  _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        for epoch in range(self.cfg.sft_epochs):
            self.model.train()
            epoch_loss  = 0.0
            steps_this_epoch = 0

            for batch_idx, batch in enumerate(self.train_loader):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                loss_mask      = batch["loss_mask"].to(self.device)

                # Targets = input shifted by 1
                targets   = input_ids[:, 1:].contiguous()
                inp       = input_ids[:, :-1].contiguous()
                loss_mask = loss_mask[:, 1:].contiguous()
                attn_mask = attention_mask[:, :-1].contiguous()

                self.optimizer.zero_grad(set_to_none=True)
                with self.autocast_ctx:
                    outputs = self.model(inp, attention_mask=attn_mask)
                    loss = masked_cross_entropy_loss(
                        outputs["logits"], targets, loss_mask
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss       += loss.item()
                steps_this_epoch += 1
                global_step      += 1

                if batch_idx % 50 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  Epoch {epoch+1} | step {batch_idx} | "
                        f"loss={loss.item():.4f} | lr={lr:.2e}"
                    )

                # Periodic mid-epoch checkpoint
                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    avg_so_far = epoch_loss / steps_this_epoch
                    print(f"  [checkpoint] step={global_step} avg_loss={avg_so_far:.4f}")
                    self.checkpointer.save(
                        self.model, self.optimizer, self.scheduler,
                        step=global_step, loss=avg_so_far,
                        extra={"epoch": epoch + 1, "stage": "sft", "mid_epoch": True},
                    )

                # Graceful stop requested by signal
                if self._stop_requested:
                    avg_so_far = epoch_loss / max(steps_this_epoch, 1)
                    print(f"  [stop] Saving checkpoint at step={global_step} loss={avg_so_far:.4f}")
                    self.checkpointer.save(
                        self.model, self.optimizer, self.scheduler,
                        step=global_step, loss=avg_so_far,
                        extra={"epoch": epoch + 1, "stage": "sft", "interrupted": True},
                    )
                    print("SFT training stopped by user.")
                    return

            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} complete — avg_loss={avg_loss:.4f}")

            if self.val_loader:
                val_loss = self.evaluate()
                print(f"  val_loss={val_loss:.4f}")

            self.checkpointer.save(
                self.model, self.optimizer, self.scheduler,
                step=global_step, loss=avg_loss,
                extra={"epoch": epoch + 1, "stage": "sft"},
            )

        print("SFT training complete.")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for batch in self.val_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            loss_mask      = batch["loss_mask"].to(self.device)

            targets   = input_ids[:, 1:].contiguous()
            inp       = input_ids[:, :-1].contiguous()
            loss_mask = loss_mask[:, 1:].contiguous()
            attn_mask = attention_mask[:, :-1].contiguous()

            with self.autocast_ctx:
                outputs = self.model(inp, attention_mask=attn_mask)
                loss = masked_cross_entropy_loss(outputs["logits"], targets, loss_mask)

            total += loss.item()
            count += 1
        return total / max(count, 1)
