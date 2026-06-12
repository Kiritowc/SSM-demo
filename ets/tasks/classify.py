"""Classification task implementation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ets.tasks.base import register_task
from ets.utils.metrics import compute_classify_metrics

CLASSIFY_MONITOR_KEYS = {
    "val_loss": "loss",
    "val_accuracy": "accuracy",
    "val_f1": "f1",
}


@register_task("classify")
class ClassifyTask:
    """Loss, metrics, and step logic for time series classification."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.num_classes = int(cfg["data"]["classify"]["num_classes"])
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, target.long())

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        pred_labels = pred.argmax(dim=-1).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        return compute_classify_metrics(target_np, pred_labels, self.num_classes)

    def training_step(
        self, model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device
    ) -> dict[str, Any]:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        loss = self.compute_loss(pred, y)
        metrics = self.compute_metrics(pred, y)
        return {"loss": loss, "metrics": metrics, "pred": pred, "target": y}

    def validation_step(
        self, model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device
    ) -> dict[str, Any]:
        with torch.no_grad():
            return self.training_step(model, batch, device)

    @property
    def monitor_name(self) -> str:
        es_cfg = self.cfg.get("train", {}).get("early_stopping", {})
        return str(es_cfg.get("monitor", "val_loss"))

    @property
    def monitor_mode(self) -> str:
        es_cfg = self.cfg.get("train", {}).get("early_stopping", {})
        return str(es_cfg.get("mode", "min"))

    def resolve_monitor_score(self, metrics: dict[str, float]) -> tuple[str, float]:
        monitor = self.monitor_name
        metric_key = CLASSIFY_MONITOR_KEYS.get(monitor)
        if metric_key is None:
            raise ValueError(
                f"Unsupported classify monitor '{monitor}'. "
                f"Choose from: {', '.join(CLASSIFY_MONITOR_KEYS)}"
            )
        if metric_key not in metrics:
            raise KeyError(f"Monitor metric '{metric_key}' missing from validation metrics.")
        return monitor, float(metrics[metric_key])

    def is_better(self, current: float, best: float) -> bool:
        if self.monitor_mode == "max":
            return current > best
        return current < best
