"""Task factory."""

from __future__ import annotations

from typing import Any

from ets.tasks.base import TASK_REGISTRY
from ets.tasks.classify import ClassifyTask  # noqa: F401 — registers classify
from ets.tasks.forecast import ForecastTask  # noqa: F401 — registers forecast


def build_task(cfg: dict[str, Any], target_scaler=None):
    """Build task handler from config."""
    task_type = cfg["task"]["type"]
    if task_type not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(f"Unknown task type: {task_type}. Available: {available}")

    cls = TASK_REGISTRY[task_type]
    if task_type == "forecast":
        return cls(cfg, target_scaler=target_scaler)
    return cls(cfg)
