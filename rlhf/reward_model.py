from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.architecture import Transformer
from model.config import ModelConfig, RLHFConfig
from training.optimizer import build_optimizer, build_lr_scheduler
from training.checkpointing import Checkpointer


class RewardModel(nn.Module):
    """Reward model for RLHF.

    Built on top of the base transformer: adds a scalar reward head
    (linear projection → scalar) that predicts how good a response is.
    The reward is extracted from the last non-padding token's hidden state.

    Stage 3 of the RLHF pipeline: (SFT model + reward head, trained on
    human preference data).
    """

    def __init__(self, base_model: Transformer, config: ModelConfig):
        super().__init__()
        self.transformer = base_model
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return scalar reward for each sequence in the batch.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)  1=token, 0=padding

        Returns:
            rewards: (batch,)
        """
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True,
        )
        # Last layer hidden states: list[-1] → (batch, seq, hidden)
        last_hidden = outputs["hidden_states"][-1]

        # Extract the last real (non-padding) token's hidden state
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1          # (batch,)
            seq_lengths = seq_lengths.clamp(min=0)
            idx = seq_lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, last_hidden.size(-1))
            last_token_hidden = last_hidden.gather(1, idx).squeeze(1)  # (batch, hidden)
        else:
            last_token_hidden = last_hidden[:, -1, :]

        reward = self.reward_head(last_token_hidden).squeeze(-1)  # (batch,)
        return reward


class RewardModelTrainer:
    """Trains the reward model on pairwise preference data.

    Loss: Bradley-Terry ranking loss
        L = -log(sigmoid(r_chosen - r_rejected))
    """

    def __init__(
        self,
        reward_model: RewardModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: RLHFConfig,
        checkpoint_dir: str = "checkpoints/reward_model",
    ):
        self.model = reward_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.checkpointer = Checkpointer(checkpoint_dir, keep_last_n=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = build_optimizer(
            reward_model,
            learning_rate=config.rm_learning_rate,
            weight_decay=0.01,
        )
        total_steps = config.rm_epochs * (
            len(train_loader) if hasattr(train_loader, "__len__") else 1000
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            warmup_steps=max(50, total_steps // 20),
            max_steps=total_steps,
        )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def _ranking_loss(
        self, chosen_reward: torch.Tensor, rejected_reward: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()
        accuracy = (chosen_reward > rejected_reward).float().mean().item()
        return loss, accuracy

    def train(self) -> None:
        print(f"Training reward model for {self.cfg.rm_epochs} epoch(s)")
        global_step = 0

        for epoch in range(self.cfg.rm_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc  = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                chosen_ids   = batch["chosen_input_ids"].to(self.device)
                chosen_mask  = batch["chosen_attention_mask"].to(self.device)
                rejected_ids  = batch["rejected_input_ids"].to(self.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with self.autocast_ctx:
                    r_chosen   = self.model(chosen_ids,   chosen_mask)
                    r_rejected = self.model(rejected_ids, rejected_mask)
                    loss, acc  = self._ranking_loss(r_chosen, r_rejected)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                epoch_acc  += acc
                global_step += 1

                if batch_idx % 50 == 0:
                    print(
                        f"  Epoch {epoch+1} | step {batch_idx} | "
                        f"loss={loss.item():.4f} | acc={acc:.3f}"
                    )

            n = len(self.train_loader)
            print(
                f"Epoch {epoch+1} — avg_loss={epoch_loss/n:.4f} | "
                f"avg_acc={epoch_acc/n:.3f}"
            )
            self.checkpointer.save(
                self.model, self.optimizer, self.scheduler,
                step=global_step, loss=epoch_loss / n,
                extra={"epoch": epoch + 1, "stage": "reward_model"},
            )

        print("Reward model training complete.")
