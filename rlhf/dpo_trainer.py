"""Direct Preference Optimization (DPO) trainer.

DPO is a simpler, more stable alternative to PPO for RLHF.
It directly optimizes the policy using preference data, without needing
a separate reward model or complex RL training loop.

Reference: https://arxiv.org/abs/2305.18290
"""
import copy
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.config import RLHFConfig
from training.optimizer import build_optimizer, build_lr_scheduler
from training.checkpointing import Checkpointer


class DPOTrainer:
    """DPO fine-tuning on (prompt, chosen, rejected) preference triplets.

    The DPO loss is:
        L_DPO = -E[log σ(β * (log π(y_w|x) - log π_ref(y_w|x))
                          - β * (log π(y_l|x) - log π_ref(y_l|x)))]

    where:
        π      = current policy (trained model)
        π_ref  = frozen reference policy (SFT model)
        y_w    = chosen response
        y_l    = rejected response
        β      = temperature controlling deviation from reference
    """

    def __init__(
        self,
        policy_model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: RLHFConfig,
        checkpoint_dir: str = "checkpoints/dpo",
    ):
        self.policy = policy_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.checkpointer = Checkpointer(checkpoint_dir, keep_last_n=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        # Frozen reference model (copy of SFT checkpoint)
        self.ref_model = copy.deepcopy(policy_model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        self.optimizer = build_optimizer(
            self.policy,
            learning_rate=config.dpo_learning_rate,
            weight_decay=0.0,
        )
        total_steps = config.dpo_epochs * (
            len(train_loader) if hasattr(train_loader, "__len__") else 1000
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            warmup_steps=max(10, total_steps // 20),
            max_steps=total_steps,
        )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def _get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sequence sum of log-probabilities for the response tokens."""
        with torch.set_grad_enabled(model is self.policy):
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"][:, :-1, :]          # (B, T-1, V)
        labels = input_ids[:, 1:].contiguous()         # (B, T-1)
        mask   = attention_mask[:, 1:].float()

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (token_log_probs * mask).sum(dim=-1)     # (B,)

    def _dpo_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        chosen_ids   = batch["chosen_input_ids"].to(self.device)
        chosen_mask  = batch["chosen_attention_mask"].to(self.device)
        rejected_ids  = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)

        with self.autocast_ctx:
            pi_chosen   = self._get_log_probs(self.policy,    chosen_ids,   chosen_mask)
            pi_rejected = self._get_log_probs(self.policy,    rejected_ids, rejected_mask)

        with torch.no_grad():
            ref_chosen   = self._get_log_probs(self.ref_model, chosen_ids,   chosen_mask)
            ref_rejected = self._get_log_probs(self.ref_model, rejected_ids, rejected_mask)

        beta = self.cfg.dpo_beta
        logits_w = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
        loss = -F.logsigmoid(logits_w).mean()

        acc  = (pi_chosen > pi_rejected).float().mean().item()
        return loss, {"loss": loss.item(), "accuracy": acc, "margin": logits_w.mean().item()}

    def train(self) -> None:
        print(f"Starting DPO training for {self.cfg.dpo_epochs} epoch(s), β={self.cfg.dpo_beta}")
        global_step = 0

        for epoch in range(self.cfg.dpo_epochs):
            self.policy.train()
            epoch_loss = 0.0
            epoch_acc  = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self._dpo_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += metrics["loss"]
                epoch_acc  += metrics["accuracy"]
                global_step += 1

                if batch_idx % 10 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  Epoch {epoch+1} | step {batch_idx} | "
                        f"loss={metrics['loss']:.4f} | "
                        f"acc={metrics['accuracy']:.3f} | "
                        f"margin={metrics['margin']:.4f} | lr={lr:.2e}"
                    )

            n = len(self.train_loader)
            print(
                f"Epoch {epoch+1} complete — avg_loss={epoch_loss/n:.4f} | "
                f"avg_acc={epoch_acc/n:.3f}"
            )
            self.checkpointer.save(
                self.policy, self.optimizer, self.scheduler,
                step=global_step, loss=epoch_loss / n,
                extra={"epoch": epoch + 1, "stage": "dpo"},
            )

        print("DPO training complete.")
