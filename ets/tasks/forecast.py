"""Forecast/regression task implementation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ets.data.scaling import inverse_transform_targets
from ets.tasks.base import register_task
from ets.utils.metrics import compute_forecast_metrics

FORECAST_MONITOR_KEYS = {
    "val_loss": "loss",
    "val_rmse": "rmse",
    "val_mae": "mae",
}


@register_task("forecast")
class ForecastTask:
    """Loss, metrics, and step logic for time series forecasting."""

    def __init__(
        self,
        cfg: dict[str, Any],
        target_scaler: StandardScaler | None = None,
    ) -> None:
        self.cfg = cfg
        self.horizon = int(cfg["data"]["horizon"])
        self.target_scaler = target_scaler
        self.criterion = nn.MSELoss()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.horizon == 1 and target.dim() == 1:
            target = target.unsqueeze(-1)
        if self.horizon > 1 and target.dim() == 2:
            target = target.unsqueeze(-1)
        return self.criterion(pred, target)

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if pred_np.ndim == 3:
            pred_np = pred_np.reshape(pred_np.shape[0], -1)
        if target_np.ndim == 1:
            target_np = target_np.reshape(-1, 1)
        elif target_np.ndim == 2 and pred_np.ndim == 2:
            pass
        else:
            target_np = target_np.reshape(pred_np.shape)

        pred_np, target_np = self._to_original_scale(pred_np, target_np)
        return compute_forecast_metrics(target_np, pred_np)

    def _to_original_scale(
        self,
        pred_np,
        target_np,
    ) -> tuple:
        pred_np = inverse_transform_targets(self.target_scaler, pred_np)
        target_np = inverse_transform_targets(self.target_scaler, target_np)
        return pred_np, target_np

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
        metric_key = FORECAST_MONITOR_KEYS.get(monitor)
        if metric_key is None:
            raise ValueError(
                f"Unsupported forecast monitor '{monitor}'. "
                f"Choose from: {', '.join(FORECAST_MONITOR_KEYS)}"
            )
        if metric_key not in metrics:
            raise KeyError(f"Monitor metric '{metric_key}' missing from validation metrics.")
        return monitor, float(metrics[metric_key])

    def is_better(self, current: float, best: float) -> bool:
        if self.monitor_mode == "max":
            return current > best
        return current < best
