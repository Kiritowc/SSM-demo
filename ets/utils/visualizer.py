"""Matplotlib-based training and prediction visualization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ets.utils.logger import log_plain, resolve_logger


def save_training_history(history: list[dict[str, Any]], path: str | Path) -> Path:
    """Save training history to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    return path


def load_training_history(path: str | Path) -> list[dict[str, Any]]:
    """Load training history from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_series(history: list[dict[str, Any]], split: str, key: str) -> list[float]:
    return [record[split][key] for record in history if key in record[split]]


def plot_training_curves(
    history: list[dict[str, Any]],
    save_dir: str | Path,
    task_type: str = "forecast",
    dpi: int = 150,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Plot training curves with matplotlib.

    Generates:
    - loss_curve.png: train/val loss
    - metrics_curve.png: task-specific metrics
    - lr_curve.png: learning rate (only if LR changes)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log = resolve_logger(logger)

    if not history:
        log.warning("Empty history, skip plotting.")
        return []

    epochs = [record["epoch"] for record in history]
    saved: list[Path] = []

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, _extract_series(history, "train", "loss"), label="train_loss", marker="o", ms=3)
    ax.plot(epochs, _extract_series(history, "val", "loss"), label="val_loss", marker="o", ms=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    loss_path = save_dir / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(loss_path, dpi=dpi)
    plt.close(fig)
    saved.append(loss_path)

    # Task metrics
    if task_type == "forecast":
        metric_keys = ["mae", "rmse"]
        metric_title = "Forecast Metrics"
    else:
        metric_keys = ["accuracy", "f1"]
        metric_title = "Classification Metrics"

    available_keys = [k for k in metric_keys if k in history[0].get("train", {})]
    if available_keys:
        fig, axes = plt.subplots(1, len(available_keys), figsize=(5 * len(available_keys), 4))
        if len(available_keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, available_keys):
            ax.plot(epochs, _extract_series(history, "train", key), label=f"train_{key}", ms=3)
            ax.plot(epochs, _extract_series(history, "val", key), label=f"val_{key}", ms=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key.upper())
            ax.set_title(key.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle(metric_title)
        metrics_path = save_dir / "metrics_curve.png"
        fig.tight_layout()
        fig.savefig(metrics_path, dpi=dpi)
        plt.close(fig)
        saved.append(metrics_path)

    # Learning rate (only when LR actually changes)
    lr_values = [record["lr"] for record in history if "lr" in record]
    if lr_values and (max(lr_values) - min(lr_values)) > 1e-12:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, lr_values, marker="o", ms=3, color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        lr_path = save_dir / "lr_curve.png"
        fig.tight_layout()
        fig.savefig(lr_path, dpi=dpi)
        plt.close(fig)
        saved.append(lr_path)

    log_plain(log, f"已保存 {len(saved)} 张图表到 {save_dir}")
    return saved


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str | Path,
    task_type: str = "forecast",
    max_samples: int = 500,
    dpi: int = 150,
    logger: logging.Logger | None = None,
) -> Path | None:
    """
    Plot prediction results on test set.

    Forecast: scatter (pred vs true) + time series overlay
    Classify: confusion-style bar or pred vs true sequence
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log = resolve_logger(logger)

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = min(len(y_true), max_samples)
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    if task_type == "classify":
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(n)
        ax.plot(x, y_true, label="true", alpha=0.8)
        ax.plot(x, y_pred, label="pred", alpha=0.8, linestyle="--")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Class Label")
        ax.set_title("Classification: True vs Predicted")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(y_true, y_pred, alpha=0.5, s=12)
        low = min(y_true.min(), y_pred.min())
        high = max(y_true.max(), y_pred.max())
        axes[0].plot([low, high], [low, high], "r--", label="ideal")
        axes[0].set_xlabel("True Value")
        axes[0].set_ylabel("Predicted Value")
        axes[0].set_title("Prediction vs Ground Truth")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        x = np.arange(n)
        axes[1].plot(x, y_true, label="true", alpha=0.8)
        axes[1].plot(x, y_pred, label="pred", alpha=0.8, linestyle="--")
        axes[1].set_xlabel("Sample Index")
        axes[1].set_ylabel("Target Value")
        axes[1].set_title(f"Forecast Overlay (first {n} samples)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    out_path = save_dir / "predictions.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    log_plain(log, f"已保存预测图到 {out_path}")
    return out_path


def visualize_training(
    history: list[dict[str, Any]],
    run_dir: str | Path,
    cfg: dict[str, Any],
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Generate all training visualizations based on config."""
    vis_cfg = cfg.get("train", {}).get("visualization", {})
    if not vis_cfg.get("enabled", True):
        return []

    run_dir = Path(run_dir)
    plot_dir = run_dir / vis_cfg.get("save_dir", "plots")
    dpi = int(vis_cfg.get("dpi", 150))
    task_type = cfg["task"]["type"]

    save_training_history(history, run_dir / "history.json")
    return plot_training_curves(
        history,
        plot_dir,
        task_type=task_type,
        dpi=dpi,
        logger=logger,
    )
