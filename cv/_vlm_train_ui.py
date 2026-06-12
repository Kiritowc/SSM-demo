"""Minimal VLM-style training console output for demos."""

from __future__ import annotations

import contextlib
import os


@contextlib.contextmanager
def suppress_training_noise():
    """静默训练过程中的 CV 验证、存盘等第三方输出。"""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def log_start(total_epochs: int) -> None:
    print("[ssdet] 开始训练（共 %d epoch）..." % total_epochs, flush=True)


def log_resume(start_epoch: int, total_epochs: int) -> None:
    print(
        "[ssdet] 从 epoch %d / %d 继续训练..." % (start_epoch, total_epochs),
        flush=True,
    )


def log_epoch_progress(epoch: int, total_epochs: int) -> None:
    done = epoch + 1
    pct = min(100, int(round(done * 100.0 / max(total_epochs, 1))))
    width = 32
    filled = int(width * done / max(total_epochs, 1))
    bar = "=" * filled + "-" * (width - filled)
    print("[ssdet] 训练  [%s] %3d%% (%d/%d)" % (bar, pct, done, total_epochs), flush=True)


def log_complete() -> None:
    print("[ssdet] 训练完成。", flush=True)


def log_release_resources() -> None:
    print("[ssdet] 释放训练资源...", flush=True)


def log_handoff_deploy() -> None:
    print("[ssdet] 训练进程结束，转入模型同步...", flush=True)


def log_camera_restart() -> None:
    print("[ssdet] 正在加载新模型...", flush=True)


def log_deploy_start() -> None:
    print("[ssdet] 正在同步模型...", flush=True)


def log_deploy_complete() -> None:
    print("[ssdet] 模型已就绪。", flush=True)


def log_deploy_failed(reason: str) -> None:
    """Demo: deploy errors are not shown on the console."""
    del reason
