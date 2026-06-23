#!/usr/bin/env python
"""CV post-training deploy: ONNX → TensorRT engine."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo

REPO_ROOT = bootstrap_repo(_REPO)

from cv.deploy.pipeline import deploy_from_config, deploy_robot_toy_engine
from cv.paths import RUNTIME_YAML
from ssm.video import start_video_server, stop_video_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--val-txt", default="")
    parser.add_argument("--yaml-path", default=RUNTIME_YAML)
    parser.add_argument("--restart-camera", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stop_video_server()
    if args.save_dir:
        ok = deploy_robot_toy_engine(
            save_dir=args.save_dir,
            model_name=args.model_name,
            val_txt=args.val_txt,
            yaml_path=args.yaml_path,
        )
    else:
        ok = deploy_from_config()
    if ok and args.restart_camera:
        print("[ssdet] 正在加载新模型...", flush=True)
        start_video_server()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
