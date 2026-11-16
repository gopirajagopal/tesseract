"""Proximal Policy Optimization (PPO) trainer for RLHF.

Implements the full RLHF-PPO pipeline:
  1. Generate responses from the current policy
  2. Score them with the reward model + KL penalty vs reference policy
  3. Compute GAE advantages
  4. Update policy with clipped PPO objective
  5. Update value function

Reference: InstructGPT / Anthropic RLHF papers.
"""
import copy
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.config import RLHFConfig
from training.optimizer import build_optimizer, build_lr_scheduler
from training.checkpointing import Checkpointer


class ValueHead(nn.Module):
    """Scalar value head appended to the transformer for PPO critic."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.proj  = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = F.tanh(self.dense(hidden_states))
        return self.proj(x).squeeze(-1)   # (batch, seq_len)


class ActorCriticModel(nn.Module):
    """Wraps the transformer with an additional value head for PPO."""

    def __init__(self, base_model: nn.Module, hidden_size: int):
        super().__init__()
        self.base  = base_model
        self.value = ValueHead(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.base(
            input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True,
        )
        logits      = outputs["logits"]                     # (B, T, V)
        last_hidden = outputs["hidden_states"][-1]          # (B, T, H)
        values      = self.value(last_hidden)               # (B, T)
        return logits, values


class PPOTrainer:
    """RLHF-PPO trainer.

    Stage 4 of the RLHF pipeline (SFT → RM → PPO-tuned policy).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        hidden_size: int,
        config: RLHFConfig,
        checkpoint_dir: str = "checkpoints/ppo",
    ):
        self.cfg = config
        self.checkpointer = Checkpointer(checkpoint_dir, keep_last_n=3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor-critic (trainable)
        self.actor_critic = ActorCriticModel(policy_model, hidden_size).to(self.device)

        # Frozen reference policy (SFT model)
        self.ref_model = copy.deepcopy(policy_model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval().to(self.device)

        # Frozen reward model
        self.reward_model = reward_model
        for p in self.reward_model.parameters():
            p.requires_grad_(False)
        self.reward_model.eval().to(self.device)

        self.optimizer = build_optimizer(
            self.actor_critic,
            learning_rate=config.ppo_learning_rate,
            weight_decay=0.0,
        )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )

    # ------------------------------------------------------------------
    # Core PPO utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token rewards = reward_model score - KL penalty."""
        # KL divergence penalty per token: log π(a|s) - log π_ref(a|s)
        ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
        ref_logits  = ref_outputs["logits"]

        log_probs     = F.log_softmax(response_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits,      dim=-1)

        actions    = input_ids[:, 1:].unsqueeze(-1)
        log_ratio  = (
            log_probs[:, :-1].gather(-1, actions).squeeze(-1)
            - ref_log_probs[:, :-1].gather(-1, actions).squeeze(-1)
        )
        kl_penalty = -self.cfg.kl_coef * log_ratio                  # (B, T-1)

        # Scalar reward from reward model → add to last token
        reward_score = self.reward_model(input_ids, attention_mask)  # (B,)
        kl_penalty[:, -1] = kl_penalty[:, -1] + reward_score

        return kl_penalty

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalised Advantage Estimation (GAE-λ)."""
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae   = torch.zeros(B, device=rewards.device)

        for t in reversed(range(T)):
            next_val = values[:, t + 1] if t < T - 1 else torch.zeros(B, device=values.device)
            delta    = rewards[:, t] + self.cfg.gamma * next_val - values[:, t]
            last_gae = delta + self.cfg.gamma * self.cfg.lam * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages * mask, returns * mask

    def _ppo_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Single PPO mini-batch update."""
        with self.autocast_ctx:
            logits, values = self.actor_critic(input_ids, attention_mask)

        actions       = input_ids[:, 1:].unsqueeze(-1)
        log_probs     = F.log_softmax(logits[:, :-1], dim=-1)
        new_log_probs = log_probs.gather(-1, actions).squeeze(-1)

        ratio = (new_log_probs - old_log_probs).exp()
        clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        value_loss = F.mse_loss(values[:, :-1], returns)
        loss = policy_loss + self.cfg.vf_coef * value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - (ratio.log())).mean().item()

        return {
            "loss":        loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "approx_kl":   approx_kl,
        }

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step: int = 0,
    ) -> Dict[str, float]:
        """Full PPO rollout + update for a single batch of prompts."""
        input_ids      = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        self.actor_critic.eval()
        with torch.no_grad(), self.autocast_ctx:
            logits, values   = self.actor_critic(input_ids, attention_mask)
            rewards          = self._compute_rewards(input_ids, attention_mask, logits)
            actions          = input_ids[:, 1:].unsqueeze(-1)
            old_log_probs    = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, actions).squeeze(-1)
            mask             = attention_mask[:, 1:].float()
            advantages, returns = self._compute_gae(rewards, values[:, :-1], mask)

        self.actor_critic.train()
        all_metrics: Dict[str, float] = {}

        for _ in range(self.cfg.ppo_epochs):
            metrics = self._ppo_step(
                input_ids, attention_mask, old_log_probs, advantages, returns
            )
            all_metrics = metrics
            if metrics["approx_kl"] > 1.5 * self.cfg.target_kl:
                break  # Early stopping if KL diverges too much

        return all_metrics

    def save(self, step: int, loss: float) -> None:
        self.checkpointer.save(
            self.actor_critic, self.optimizer, None,
            step=step, loss=loss, extra={"stage": "ppo"},
        )
