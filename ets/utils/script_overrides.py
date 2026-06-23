"""Shared CLI-to-YAML override helpers for training scripts."""

from __future__ import annotations

FORECAST_MONITORS = ("val_loss", "val_mse", "val_rmse", "val_mae")
CLASSIFY_MONITORS = ("val_loss", "val_accuracy", "val_f1")
MAX_MONITORS = ("val_accuracy", "val_f1")


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def resolve_early_stopping(task: str, monitor: str) -> tuple[str, str]:
    """Return a task-valid monitor name and early-stopping mode."""
    if task == "classify":
        if monitor not in CLASSIFY_MONITORS:
            monitor = "val_accuracy"
        mode = "max" if monitor in MAX_MONITORS else "min"
    else:
        if monitor not in FORECAST_MONITORS:
            monitor = "val_rmse"
        mode = "min"
    return monitor, mode


def build_train_overrides(args) -> list[str]:
    """Map train/backtest CLI args onto YAML overrides (CLI wins over YAML)."""
    overrides = [
        f"data.profile={args.data_profile}",
        f"model.name={args.model}",
        f"model.hidden_size={args.hidden_size}",
        f"model.num_layers={args.num_layers}",
        f"model.dropout={args.dropout}",
        f"model.bidirectional={bool_str(args.bidirectional)}",
        f"task.type={args.task}",
        f"data.window_size={args.window_size}",
        f"data.horizon={args.horizon}",
        f"data.scale_target={bool_str(args.scale_target)}",
        f"train.epochs={args.epochs}",
        f"train.batch_size={args.batch_size}",
        f"train.lr={args.lr}",
        f"train.weight_decay={args.weight_decay}",
        f"train.optimizer={args.optimizer}",
        f"train.num_workers={args.num_workers}",
        f"seed={args.seed}",
        f"device={args.device}",
    ]
    if hasattr(args, "log_interval"):
        overrides.append(f"train.log_interval={args.log_interval}")
    if hasattr(args, "early_stopping"):
        overrides.extend(
            [
                f"train.early_stopping.enabled={bool_str(args.early_stopping)}",
                f"train.early_stopping.patience={args.patience}",
                f"train.early_stopping.monitor={args.monitor}",
                f"train.early_stopping.mode={args.monitor_mode}",
            ]
        )
    if getattr(args, "exp_name", ""):
        overrides.append(f"train.exp_name={args.exp_name}")
    if getattr(args, "resume", ""):
        overrides.append(f"train.resume={args.resume}")
    if getattr(args, "scheduler", ""):
        overrides.append(f"train.scheduler={args.scheduler}")
    if getattr(args, "num_channels", ""):
        overrides.append(f"model.num_channels={args.num_channels}")
    if getattr(args, "dilations", ""):
        overrides.append(f"model.dilations={args.dilations}")
    if getattr(args, "encoder_hidden_size", None) is not None:
        overrides.append(f"model.encoder_hidden_size={args.encoder_hidden_size}")
    if getattr(args, "decoder_hidden_size", None) is not None:
        overrides.append(f"model.decoder_hidden_size={args.decoder_hidden_size}")
    if getattr(args, "decoder_layers", None) is not None:
        overrides.append(f"model.decoder_layers={args.decoder_layers}")
    if getattr(args, "log_dir", ""):
        overrides.append(f"train.log_dir={args.log_dir}")
    if getattr(args, "input_mode", None):
        overrides.append(f"data.input_mode={args.input_mode}")
    if getattr(args, "kernel_size", None):
        overrides.append(f"model.kernel_size={args.kernel_size}")
    if getattr(args, "no_visualization", False):
        overrides.append("train.visualization.enabled=false")
    return overrides


def build_infer_overrides(args) -> list[str]:
    """Map inference script args onto YAML override strings."""
    return [
        f"train.batch_size={args.batch_size}",
        f"train.num_workers={args.num_workers}",
        f"device={args.device}",
    ]
