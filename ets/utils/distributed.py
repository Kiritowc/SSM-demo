"""Distributed training utilities (skeleton for future DDP support)."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Check if distributed training is enabled."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def setup_distributed(local_rank: int = -1) -> int:
    """
    Initialize distributed training if environment variables are set.

    Returns local rank (-1 for single-process training).
    """
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if local_rank < 0:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank
    return -1


def cleanup_distributed() -> None:
    """Destroy distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def wrap_model_ddp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    """Wrap model with DistributedDataParallel if distributed is enabled."""
    if is_distributed() and local_rank >= 0:
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
        )
    return model
