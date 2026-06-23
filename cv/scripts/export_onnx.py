#!/usr/bin/env python
"""CV ONNX export entry point."""

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

from cv.core.exporting import convert_and_save_model
from cv.paths import ROBOT_TOY_BEST_BIN, ROBOT_TOY_RUN_BIN, RUNTIME_YAML, SAMPLE_IMAGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--yaml", type=str, default=RUNTIME_YAML)
    parser.add_argument("--model", type=str, default="ssg_a")
    parser.add_argument("--spp", type=str, default="spp")
    parser.add_argument("--weight", type=str, default=ROBOT_TOY_BEST_BIN)
    parser.add_argument("--save_path", type=str, default=ROBOT_TOY_RUN_BIN)
    parser.add_argument("--img", type=str, default=SAMPLE_IMAGE)
    parser.add_argument("--thresh", type=float, default=0.50)
    parser.add_argument("--ins", type=str, default=None)
    parser.add_argument("--ous", type=str, default=None)
    parser.add_argument("--plain-onnx", type=str, default=None)
    parser.add_argument("--skip-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    convert_and_save_model(parse_args())


if __name__ == "__main__":
    main()
