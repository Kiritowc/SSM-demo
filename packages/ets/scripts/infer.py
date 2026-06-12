#!/usr/bin/env python
"""Inference entry point."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ets.engine.predictor import Predictor
from ets.utils.checkpoint import resolve_checkpoint
from ets.utils.config import load_config
from ets.utils.logger import setup_logger
from ets.utils.script_overrides import build_infer_overrides


def main() -> None:
    args = parse_args()
    logger = setup_logger("ets")

    if not args.checkpoint:
        raise ValueError("请在 infer.py 底部 parse_args() 设置 --checkpoint 为训练产出的 run 目录")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    checkpoint = resolve_checkpoint(ckpt_path, prefer="best")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = deepcopy(state["cfg"])
    cfg["device"] = args.device
    cfg["train"]["batch_size"] = args.batch_size
    cfg["train"]["num_workers"] = args.num_workers

    if args.merge_data_yaml:
        config_path = PROJECT_ROOT / args.config
        yaml_cfg = load_config(
            [config_path],
            overrides=build_infer_overrides(args),
            project_root=PROJECT_ROOT,
        )
        cfg["data"] = yaml_cfg["data"]
        logger.info("数据配置来自 %s，模型结构来自 checkpoint", config_path)

    output_path = PROJECT_ROOT / args.output

    predictor = Predictor(
        checkpoint_path=checkpoint,
        cfg=cfg,
        device=args.device,
        project_root=str(PROJECT_ROOT),
    )

    saved_path = predictor.save_predictions_csv(output_path)
    logger.info("推理完成，输出: %s", saved_path)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="ETS 批量推理")
    parser.add_argument(
        "--config",
        default="../../configs/ets/default.yaml",
        help="仅 merge_data_yaml 时用于覆盖数据段",
    )
    parser.add_argument("--checkpoint", default="", help="run 目录或 weights/best.pt，训练后填写")
    parser.add_argument("--output", default="outputs/predictions.csv", help="预测结果输出 CSV")
    parser.add_argument("--device", default="cuda", help="推理设备")
    parser.add_argument("--batch-size", default=64, type=int, help="推理 DataLoader batch")
    parser.add_argument("--num-workers", default=0, type=int, help="推理 DataLoader workers")
    parser.add_argument(
        "--merge-data-yaml",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="是否用 YAML 的 data 段覆盖 checkpoint 中的数据配置",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
