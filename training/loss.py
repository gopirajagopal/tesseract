import torch
import torch.nn.functional as F
from typing import Optional


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard next-token prediction cross-entropy loss.

    Args:
        logits:  (batch, seq_len, vocab_size)
        targets: (batch, seq_len)  — target token IDs

    Returns:
        Scalar loss.
    """
    # Flatten to (batch*seq_len, vocab_size) and (batch*seq_len,)
    logits  = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss with a binary mask — used for SFT.

    Only tokens where loss_mask == 1 contribute to the loss.

    Args:
        logits:    (batch, seq_len, vocab_size)
        targets:   (batch, seq_len)
        loss_mask: (batch, seq_len) float tensor of 0/1

    Returns:
        Scalar mean loss over unmasked positions.
    """
    B, T, V = logits.shape
    logits_flat  = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    mask_flat    = loss_mask.view(B * T)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    return loss


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from a cross-entropy loss value."""
    return torch.exp(loss)
