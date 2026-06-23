"""Evaluation metrics for forecast and classification tasks."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def format_params(num_params: int) -> str:
    """Format parameter count as K/M abbreviation."""
    if num_params >= 1_000_000:
        value = num_params / 1_000_000
        text = f"{value:.1f}M"
        return text.replace(".0M", "M")
    return f"{round(num_params / 1000)}K"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1_scores = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall + 1e-8))
    if not f1_scores:
        return 0.0
    return float(np.mean(f1_scores))


def compute_classify_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, num_classes),
    }
