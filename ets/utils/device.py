"""Device selection utilities."""

from __future__ import annotations

import torch


def get_device(device_cfg: str = "auto") -> torch.device:
    """Resolve torch device from config string."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_cfg)
