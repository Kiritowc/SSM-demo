#!/usr/bin/env python3
"""SSM platform CLI — service lifecycle (up / down)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from ssm.paths import repo_root
from ssm.video import start_video_server, stop_video_server


def _run(cmd: list[str], *, cwd=None) -> int:
    root = repo_root()
    env = os.environ.copy()
    env["SSM_ROOT"] = str(root)
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd or root), env=env)


def cmd_up(args: argparse.Namespace) -> int:
    root = repo_root()
    if args.vlm_only and args.video_only:
        print("error: --vlm-only and --video-only are mutually exclusive", file=sys.stderr)
        return 2

    start_vlm = not args.video_only
    start_video = not args.vlm_only

    rc = 0
    if start_vlm:
        rc = _run(["bash", str(root / "vlm/services/scripts/start.sh")])
    if rc == 0 and start_video:
        print("+ video server (background)", flush=True)
        start_video_server()
    return rc


def cmd_down(_: argparse.Namespace) -> int:
    _run(["bash", str(repo_root() / "vlm/services/scripts/stop.sh")])
    stop_video_server(wait_sec=0)
    return 0


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("SSM_ROOT", str(repo_root()))
    parser = argparse.ArgumentParser(prog="ssm", description="SSM service CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="Start services (default: VLM + video)")
    up.add_argument("--vlm-only", action="store_true", help="Start VLM only")
    up.add_argument("--video-only", action="store_true", help="Start video server only")
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down", help="Stop services")
    down.set_defaults(func=cmd_down)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
