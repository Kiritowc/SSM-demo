"""Camera helpers for post-training engine deploy (VLM stays running)."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(rel_path: str) -> None:
    script = REPO_ROOT / rel_path
    if not script.is_file():
        return
    subprocess.run(["bash", str(script)], cwd=str(REPO_ROOT), check=False)


def stop_camera(*, wait_sec: float = 2.0) -> None:
    _run_script("camera/scripts/stop_camera.sh")
    if wait_sec > 0:
        time.sleep(wait_sec)


def start_camera(*, wait_sec: float = 3.0) -> None:
    script = REPO_ROOT / "services/video/server.py.sh"
    if not script.is_file():
        return
    log_path = REPO_ROOT / "camera/logs/camera-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.Popen(
            ["bash", str(script)],
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if wait_sec > 0:
        time.sleep(wait_sec)


def ensure_single_train_process() -> None:
    me = os.getpid()
    try:
        out = subprocess.run(
            ["pgrep", "-f", r"train\.py"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        pid = int(line)
        if pid == me:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(1)
