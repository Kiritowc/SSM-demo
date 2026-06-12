#!/usr/bin/env python
"""ONNX export entry point."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        ckpt_path = PROJECT_ROOT / ckpt_path
    checkpoint = str(resolve_checkpoint(ckpt_path, prefer="best"))

    output_path = PROJECT_ROOT / args.output

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

    parser = argparse.ArgumentParser(description="ETS ONNX 导出")
    parser.add_argument("--checkpoint", default="", help="run 目录或 weights/best.pt，训练后填写")
    parser.add_argument("--output", default="exports/model.onnx", help="ONNX 模型输出路径")
    parser.add_argument("--opset-version", default=17, type=int, help="ONNX opset 版本")
    parser.add_argument("--device", default="cpu", help="导出时使用的设备")
    return parser.parse_args()


if __name__ == "__main__":
    main()
