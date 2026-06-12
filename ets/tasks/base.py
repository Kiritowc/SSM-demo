"""Task protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

TASK_REGISTRY: dict[str, type] = {}


def register_task(name: str):
    """Decorator to register a task implementation."""

    def decorator(cls: type) -> type:
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


@runtime_checkable
class TaskProtocol(Protocol):
    """Interface for task-specific training and monitoring logic."""

    cfg: dict[str, Any]

    def training_step(
        self, model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device
    ) -> dict[str, Any]: ...

    def validation_step(
        self, model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device
    ) -> dict[str, Any]: ...

    def resolve_monitor_score(self, metrics: dict[str, float]) -> tuple[str, float]: ...

    def is_better(self, current: float, best: float) -> bool: ...

    @property
    def monitor_mode(self) -> str: ...
