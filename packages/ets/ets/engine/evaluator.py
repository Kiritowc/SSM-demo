"""Evaluation engine for validation and test sets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from ets.data.scaling import inverse_transform_targets
from ets.tasks.factory import build_task
from ets.utils.logger import log_split_metrics, resolve_logger


class Evaluator:
    """Evaluate model on a dataloader."""

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: dict[str, Any],
        device: torch.device,
        target_scaler=None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.task = build_task(cfg, target_scaler=target_scaler)
        self.logger = resolve_logger(logger)

    def evaluate(self, dataloader, split_name: str = "test") -> dict[str, float]:
        """Run evaluation and return averaged metrics."""
        self.model.eval()
        metric_sums: dict[str, float] = {}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                result = self.task.validation_step(self.model, batch, self.device)
                for key, value in result["metrics"].items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                num_batches += 1

        metrics = {k: v / max(num_batches, 1) for k, v in metric_sums.items()}
        log_split_metrics(self.logger, split_name, metrics)
        return metrics

    def predict(self, dataloader) -> dict[str, np.ndarray]:
        """Run inference and return predictions and targets."""
        self.model.eval()
        preds_list = []
        targets_list = []
        task_type = self.cfg["task"]["type"]

        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"]
                pred = self.model(x)
                if task_type == "classify":
                    pred = pred.argmax(dim=-1)
                preds_list.append(pred.cpu().numpy())
                targets_list.append(y.numpy())

        predictions = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        if task_type == "forecast":
            target_scaler = getattr(self.task, "target_scaler", None)
            predictions = inverse_transform_targets(target_scaler, predictions)
            targets = inverse_transform_targets(target_scaler, targets)

        return {
            "predictions": predictions,
            "targets": targets,
        }
