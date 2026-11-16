"""Distributed training utilities for multi-GPU / multi-node training."""
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_distributed(backend: str = "nccl") -> None:
    """Initialise the default process group for DDP training.

    Reads RANK, LOCAL_RANK, and WORLD_SIZE from environment variables
    (set automatically by torchrun / deepspeed launch).
    """
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        print(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        print("Single-process training (no DDP).")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: nn.Module, device_ids: Optional[list] = None) -> nn.Module:
    """Wrap model with DistributedDataParallel if running multi-GPU."""
    if not (dist.is_available() and dist.is_initialized()):
        return model
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )


def is_main_process() -> bool:
    """Return True only for rank 0 (or single-process runs)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all ranks."""
    if get_world_size() <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / get_world_size()
