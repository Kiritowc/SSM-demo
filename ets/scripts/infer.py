#!/usr/bin/env python
"""Inference entry point."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo

REPO_ROOT = bootstrap_repo(_REPO)

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
        ckpt_path = REPO_ROOT / ckpt_path
    checkpoint = resolve_checkpoint(ckpt_path, prefer="best")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = deepcopy(state["cfg"])
    cfg["device"] = args.device
    cfg["train"]["batch_size"] = args.batch_size
    cfg["train"]["num_workers"] = args.num_workers

    if args.merge_data_yaml:
        config_path = REPO_ROOT / args.config
        yaml_cfg = load_config(
            [config_path],
            overrides=build_infer_overrides(args),
            project_root=REPO_ROOT,
        )
        cfg["data"] = yaml_cfg["data"]
        logger.info("数据配置来自 %s，模型结构来自 checkpoint", config_path)

    output_path = REPO_ROOT / args.output

    predictor = Predictor(
        checkpoint_path=checkpoint,
        cfg=cfg,
        device=args.device,
        project_root=str(REPO_ROOT),
    )

    saved_path = predictor.save_predictions_csv(output_path)
    logger.info("推理完成，输出: %s", saved_path)


def parse_args():
    import argparse

    from ets.utils.argparse_compat import ensure_boolean_optional_action

    ensure_boolean_optional_action()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--merge-data-yaml", default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == "__main__":
    main()
