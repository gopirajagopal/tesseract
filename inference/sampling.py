"""Sampling strategies for autoregressive text generation."""
from typing import Optional

import torch
import torch.nn.functional as F


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy decoding — always pick the highest-probability token."""
    return logits.argmax(dim=-1)


def temperature_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample from logits after scaling by temperature.

    temperature → 0  : approaches greedy
    temperature = 1  : standard sampling
    temperature > 1  : more random / creative
    """
    if temperature <= 0.0:
        return greedy(logits)
    scaled = logits / temperature
    probs  = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_k_sample(logits: torch.Tensor, top_k: int, temperature: float = 1.0) -> torch.Tensor:
    """Top-K sampling: restrict to the K most likely tokens before sampling."""
    if top_k <= 0:
        return temperature_sample(logits, temperature)

    top_k = min(top_k, logits.size(-1))
    kth_val = logits.topk(top_k, dim=-1).values[..., -1, None]
    filtered = logits.masked_fill(logits < kth_val, float("-inf"))
    return temperature_sample(filtered, temperature)


def top_p_sample(
    logits: torch.Tensor,
    top_p: float,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Nucleus (top-p) sampling: keep the smallest set of tokens whose
    cumulative probability ≥ top_p.

    Reference: https://arxiv.org/abs/1904.09751
    """
    if top_p >= 1.0:
        return temperature_sample(logits, temperature)

    scaled = logits / max(temperature, 1e-8)
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative prob above the threshold
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    # Keep at least min_tokens_to_keep
    sorted_mask[..., :min_tokens_to_keep] = False

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # Scatter back to original token ordering
    filtered_logits = torch.zeros_like(scaled).scatter_(-1, sorted_indices, sorted_logits)
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_k_top_p_sample(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combined top-k + top-p + temperature sampling with repetition penalty.

    This is the default sampling strategy used during generation.
    """
    if repetition_penalty != 1.0 and input_ids is not None:
        logits = _apply_repetition_penalty(logits, input_ids, repetition_penalty)

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = logits.topk(top_k, dim=-1).values[..., -1, None]
        logits  = logits.masked_fill(logits < kth_val, float("-inf"))

    if top_p < 1.0:
        return top_p_sample(logits, top_p=top_p, temperature=temperature)

    return temperature_sample(logits, temperature)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalise tokens that have already appeared in the sequence."""
    score = logits.gather(-1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(-1, input_ids, score)
    return logits
