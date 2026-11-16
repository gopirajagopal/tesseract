import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA).

    A generalisation of Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
    Query heads are split into groups; each group shares one KV head.
    This significantly reduces KV-cache size during inference.

    Reference: https://arxiv.org/abs/2305.13245
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_flash = config.use_flash_attention
        self.attn_dropout = config.attention_dropout

        self.scale = self.head_dim ** -0.5

        # Separate projections (no bias — modern LLMs)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

        self.rotary = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            scaling_factor=config.rope_scaling,
        )

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand KV heads to match the number of Q heads."""
        if n_rep == 1:
            return x
        b, num_kv_heads, seq, hd = x.shape
        x = x[:, :, None, :, :].expand(b, num_kv_heads, n_rep, seq, hd)
        return x.reshape(b, num_kv_heads * n_rep, seq, hd)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states:   (batch, seq_len, hidden_size)
            attention_mask:  (batch, 1, seq_len, seq_len) additive mask; -inf for masked positions
            position_ids:    (batch, seq_len)
            past_key_value:  cached (k, v) tensors from previous forward passes
            use_cache:       whether to return updated KV cache

        Returns:
            attn_output:     (batch, seq_len, hidden_size)
            past_key_value:  updated cache if use_cache else None
        """
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (batch, heads, seq, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rotary(q, k, position_ids)

        # Append to KV cache if provided
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v) if use_cache else None

        # Expand KV heads to match Q heads (GQA)
        k = self._repeat_kv(k, self.num_kv_groups).contiguous()
        v = self._repeat_kv(v, self.num_kv_groups).contiguous()

        # Attention
        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            # PyTorch 2.0+ Flash Attention (fused kernel)
            # Use is_causal flag instead of explicit mask to avoid contiguity issues.
            # When past_key_value is not None we are in KV-cache single-token mode:
            # the new token must attend to ALL past keys, so is_causal=False.
            dropout_p = self.attn_dropout if self.training else 0.0
            use_causal = past_key_value is None
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=use_causal,
            )
        else:
            attn_output = self._manual_attention(q, k, v, attention_mask)

        # Merge heads → (batch, seq, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Causal mask if none provided
        if mask is None:
            T = q.shape[2]
            causal = torch.triu(
                torch.full((T, T), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1
            )
            scores = scores + causal
        else:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        if self.attn_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout)

        return torch.matmul(attn_weights, v)
