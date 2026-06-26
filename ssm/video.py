"""Video + web demo server lifecycle (SSDet + VLM integration)."""

from __future__ import annotations

import subprocess
import sys
import time

from ssm.paths import get_paths, repo_root

VIDEO_SERVER = "apps/video/server.py"


def video_server_path():
    return repo_root() / VIDEO_SERVER


def video_server_log_path():
    return get_paths().apps_video / "logs" / "video-server.log"


def stop_video_server(*, wait_sec: float = 2.0) -> None:
    subprocess.call(["pkill", "-f", VIDEO_SERVER], stderr=subprocess.DEVNULL)
    if wait_sec > 0:
        time.sleep(wait_sec)


def start_video_server(*, wait_sec: float = 3.0) -> None:
    server = video_server_path()
    if not server.is_file():
        return
    log_path = video_server_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.Popen(
            [sys.executable, str(server)],
            cwd=str(repo_root()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if wait_sec > 0:
        time.sleep(wait_sec)
