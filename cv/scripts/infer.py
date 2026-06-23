#!/usr/bin/env python
"""CV image inference entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo

REPO_ROOT = bootstrap_repo(_REPO)

from cv.inference import ssDet
from cv.paths import ROBOT_TOY_RUN_BIN, RUNTIME_NAMES, SAMPLE_IMAGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=str, default=SAMPLE_IMAGE)
    parser.add_argument("--conf", default=0.5, type=float)
    parser.add_argument("--nms", default=0.5, type=float)
    parser.add_argument("--weight", type=str, default=ROBOT_TOY_RUN_BIN)
    parser.add_argument("--names", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default="outputs/inference-result.jpg")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names_path = args.names or RUNTIME_NAMES
    src_path = Path(args.source).expanduser()
    if not src_path.is_file():
        raise SystemExit("image not found: %s" % src_path.resolve())
    srcimg = cv2.imread(str(src_path))
    if srcimg is None or srcimg.size == 0:
        raise SystemExit("opencv could not read image: %s" % src_path.resolve())

    model = ssDet(conf=args.conf, nms=args.nms, weight=args.weight, names=names_path)
    _res_data, annotated = model.detect(srcimg)
    if not args.no_save:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), annotated):
            raise SystemExit("failed to write output: %s" % out_path)
        print("wrote: %s" % out_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
