import torch
import torch.nn as nn
from typing import Optional, Tuple


class TokenEmbedding(nn.Module):
    """Standard learned token embedding."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes position information directly into attention query/key vectors
    via rotation in 2D subspaces. Enables better length generalization
    than learned absolute position embeddings.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
        scaling_factor: Optional[float] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        theta = self.theta
        if self.scaling_factor is not None:
            theta = theta * self.scaling_factor

        dev = device or torch.device("cpu")

        # Compute inverse frequencies for even-indexed dimensions
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=dev) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache for positions 0..seq_len-1
        t = torch.arange(seq_len, dtype=torch.float32, device=dev)
        freqs = torch.outer(t, inv_freq)                 # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)          # (seq_len, head_dim)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension to the first half with negation."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_kv_heads, seq_len, head_dim)
            position_ids: optional (batch, seq_len) position indices

        Returns:
            Rotated (q, k) tensors of same shape.
        """
        seq_len = q.shape[2]

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len, device=q.device)
            self.max_seq_len = seq_len

        if position_ids is None:
            cos = self.cos_cache[:, :, :seq_len, :]
            sin = self.sin_cache[:, :, :seq_len, :]
        else:
            cos = self.cos_cache[0, 0][position_ids].unsqueeze(1)  # (batch, 1, seq, dim)
            sin = self.sin_cache[0, 0][position_ids].unsqueeze(1)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot.type_as(q), k_rot.type_as(k)
