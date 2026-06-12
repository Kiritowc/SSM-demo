"""Structured logging utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

DEFAULT_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
TRAIN_FMT = "%(asctime)s  %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
METRIC_VALUE_WIDTH = 14


class TrainFormatter(logging.Formatter):
    """Train-style formatter: plain lines skip timestamp."""

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "plain", False):
            return record.getMessage()
        return super().format(record)


def _metric_width() -> int:
    return METRIC_VALUE_WIDTH


def _sorted_metric_keys(metrics: dict[str, float], include_loss: bool) -> list[str]:
    keys = sorted(metrics)
    if not include_loss:
        keys = [key for key in keys if key != "loss"]
    elif "loss" in keys:
        keys = ["loss"] + [key for key in keys if key != "loss"]
    return keys


def format_metric_fields(
    metrics: dict[str, float],
    *,
    include_loss: bool = False,
    value_width: int | None = None,
) -> str:
    """Format metric key/value pairs into an aligned string."""
    width = value_width or _metric_width()
    parts = []
    for key in _sorted_metric_keys(metrics, include_loss=include_loss):
        label = key.upper() if key != "loss" else "Loss"
        parts.append(f"{label}: {metrics[key]:{width}.6f}")
    return ", ".join(parts)


def format_epoch_line(
    epoch: int,
    total: int,
    train_loss: float,
    val_loss: float,
) -> str:
    """Format a compact single-line epoch log without timestamp."""
    epoch_width = len(str(total))
    return (
        f"Epoch {epoch:>{epoch_width}d}/{total}, "
        f"Train Loss: {train_loss:.6f}, "
        f"Val Loss: {val_loss:.6f}"
    )


def format_metric_lines(
    metrics: dict[str, float],
    *,
    include_loss: bool = False,
) -> list[str]:
    """Format compact metric lines, one metric per row."""
    lines = []
    for key in _sorted_metric_keys(metrics, include_loss=include_loss):
        label = "Loss" if key == "loss" else key.upper()
        lines.append(f"{label}: {metrics[key]:.6f}")
    return lines


def log_plain(logger: logging.Logger, message: str) -> None:
    """Log a line without timestamp."""
    logger.info(message, extra={"plain": True})


def log_timestamped(logger: logging.Logger, message: str) -> None:
    """Log a single line with timestamp."""
    logger.info(message)


def log_eval_metrics(logger: logging.Logger, metrics: dict[str, float]) -> None:
    """Log eval metrics after epoch line as compact plain lines."""
    for line in format_metric_lines(metrics, include_loss=False):
        log_plain(logger, line)


def log_split_metrics(
    logger: logging.Logger,
    split_name: str,
    metrics: dict[str, float],
    *,
    include_loss: bool = False,
) -> None:
    """Log final split metrics as compact plain lines."""
    lines = format_metric_lines(metrics, include_loss=include_loss)
    if not lines:
        return
    log_plain(logger, split_name)
    for line in lines:
        log_plain(logger, line)


def resolve_logger(
    logger: logging.Logger | None = None,
    name: str = "ets",
    style: str = "train",
) -> logging.Logger:
    """Return an existing logger or create a single shared console logger."""
    if logger is not None:
        return logger
    existing = logging.getLogger(name)
    if existing.handlers:
        return existing
    return setup_logger(name, style=style)


def setup_logger(
    name: str = "ets",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    style: str = "default",
) -> logging.Logger:
    """Create a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    if style == "train":
        formatter: logging.Formatter = TrainFormatter(fmt=TRAIN_FMT, datefmt=DATE_FMT)
    else:
        formatter = logging.Formatter(fmt=DEFAULT_FMT, datefmt=DATE_FMT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
