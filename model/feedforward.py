import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Uses the gated linear unit variant with Swish (SiLU) activation.
    Outperforms vanilla FFN with GELU/ReLU on language modeling tasks.
    Used in PaLM, LLaMA, and Claude-style architectures.

    SwiGLU(x) = SiLU(W1*x) ⊙ (W3*x)
    output    = W2 * SwiGLU(x)

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)
        self.dropout   = nn.Dropout(config.hidden_dropout) if config.hidden_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate branch: SiLU(W1*x)  |  Up branch: W3*x
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        hidden = self.dropout(gate * up)
        return self.down_proj(hidden)
