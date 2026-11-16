import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> AdamW:
    """Build AdamW optimizer with weight decay applied only to non-bias/norm params.

    Following the convention in GPT/LLaMA training:
    - Weight decay is applied to weight matrices (2D+)
    - No weight decay on biases, LayerNorm/RMSNorm weights, embeddings
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower() or "bias" in name.lower() or "embed" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        fused=_fused_adamw_available(),
    )
    return optimizer


def _fused_adamw_available() -> bool:
    """Check if PyTorch fused AdamW (faster CUDA kernel) is available."""
    if not torch.cuda.is_available():
        return False
    try:
        import inspect
        return "fused" in inspect.signature(AdamW.__init__).parameters
    except Exception:
        return False


def build_lr_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine decay with linear warmup.

    LR schedule used in most modern LLM training runs:
    - Linear warmup from 0 → peak LR over `warmup_steps`
    - Cosine decay from peak LR → min LR over the rest of training
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)
