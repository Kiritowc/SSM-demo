#!/usr/bin/env python3
"""SSM platform CLI."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    return Path(os.environ.get("SSM_ROOT", Path(__file__).resolve().parents[1]))


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    env = os.environ.copy()
    env["SSM_ROOT"] = str(_root())
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd or _root()), env=env)


def cmd_up(args: argparse.Namespace) -> int:
    root = _root()
    rc = 0
    if args.vlm:
        rc = _run(["bash", str(root / "services/vlm-server/scripts/start.sh")])
    if rc == 0 and args.video:
        rc = _run([sys.executable, str(root / "services/video/server.py")], cwd=root)
    return rc


def cmd_down(args: argparse.Namespace) -> int:
    root = _root()
    _run(["bash", str(root / "services/vlm-server/scripts/stop.sh")])
    subprocess.call(["pkill", "-f", "services/video/server.py"], stderr=subprocess.DEVNULL)
    return 0


def cmd_demo(_: argparse.Namespace) -> int:
    root = _root()
    _run(["bash", str(root / "services/vlm-server/scripts/start.sh")])
    print("Web UI: http://127.0.0.1:9080/")
    print("Start video in another terminal: ssm up --video-only")
    return 0


def cmd_train_cv(args: argparse.Namespace) -> int:
    root = _root()
    cmd = [sys.executable, "-m", "sunshink_cv.train"]
    if args.configfile:
        cmd += ["--configfile", args.configfile]
    rc = _run(cmd)
    if rc == 0 and args.deploy:
        return _run([sys.executable, "-m", "sunshink_cv.post_deploy", "--restart-camera"])
    return rc


def cmd_train_ets(args: argparse.Namespace) -> int:
    root = _root()
    script = root / "packages/ets/scripts/train.py"
    cmd = [sys.executable, str(script), "--config", str(root / "configs/ets/default.yaml")]
    if args.model:
        cmd += ["--model", args.model]
    if args.data:
        cmd += ["--data-profile", args.data]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    return _run(cmd)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("SSM_ROOT", str(_root()))
    parser = argparse.ArgumentParser(prog="ssm", description="SSM monorepo CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="Start services")
    up.add_argument("--vlm-only", action="store_true", dest="vlm")
    up.add_argument("--video-only", action="store_true", dest="video")
    up.set_defaults(vlm=True, video=True, func=cmd_up)

    down = sub.add_parser("down", help="Stop services")
    down.set_defaults(func=cmd_down)

    demo = sub.add_parser("demo", help="Start VLM and print demo URLs")
    demo.set_defaults(func=cmd_demo)

    train = sub.add_parser("train", help="Train models")
    train_sub = train.add_subparsers(dest="target", required=True)

    cv = train_sub.add_parser("cv", help="Train CV model")
    cv.add_argument("--configfile", default="")
    cv.add_argument("--deploy", action="store_true")
    cv.set_defaults(func=cmd_train_cv)

    ets = train_sub.add_parser("ets", help="Train ETS model")
    ets.add_argument("--model", default="ets_a")
    ets.add_argument("--data", default="scada")
    ets.add_argument("--epochs", type=int, default=0)
    ets.set_defaults(func=cmd_train_ets)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
