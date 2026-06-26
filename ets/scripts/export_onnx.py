#!/usr/bin/env python
"""ONNX export entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo, ensure_runtime_python

REPO_ROOT = bootstrap_repo(_REPO)
ensure_runtime_python()

from ets.export.onnx import export_onnx
from ets.utils.checkpoint import resolve_checkpoint
from ets.utils.logger import setup_logger


def main() -> None:
    args = parse_args()
    logger = setup_logger("ets")

    if not args.checkpoint:
        raise ValueError(
            "请在 export_onnx.py 底部 parse_args() 设置 --checkpoint 为训练产出的 run 目录"
        )
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    checkpoint = str(resolve_checkpoint(ckpt_path, prefer="best"))

    output_path = REPO_ROOT / args.output

    export_onnx(
        checkpoint_path=checkpoint,
        output_path=output_path,
        cfg=None,
        opset_version=args.opset_version,
        device=args.device,
    )
    logger.info("导出完成: %s", output_path)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="exports/model.onnx")
    parser.add_argument("--opset-version", default=17, type=int)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
