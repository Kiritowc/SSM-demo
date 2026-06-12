#!/usr/bin/env python
"""Walk-forward backtest entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ets.engine.backtest import walk_forward_evaluate
from ets.models.registry import MODEL_REGISTRY
from ets.utils.config import load_config
from ets.utils.logger import setup_logger
from ets.utils.script_overrides import build_train_overrides


def main() -> None:
    args = parse_args()
    logger = setup_logger("ets")
    config_path = PROJECT_ROOT / args.config
    overrides = build_train_overrides(args)
    overrides.extend(
        [
            "eval.mode=walk_forward",
            f"eval.n_splits={args.n_splits}",
            f"eval.train_window={args.train_window or 'null'}",
            f"eval.fold_epochs={args.fold_epochs}",
        ]
    )
    cfg = load_config([config_path], overrides=overrides, project_root=PROJECT_ROOT)

    logger.info(
        "回测: 模型=%s | 任务=%s | fold_epochs=%d | n_splits=%d",
        cfg["model"]["name"],
        cfg["task"]["type"],
        args.fold_epochs,
        args.n_splits,
    )

    result = walk_forward_evaluate(cfg, project_root=PROJECT_ROOT)
    logger.info(
        "Walk-forward 完成: MAE=%.4f RMSE=%.4f MAPE=%.2f%% (%d folds)",
        result["mae_mean"],
        result["rmse_mean"],
        result["mape_mean"],
        len(result["folds"]),
    )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("结果已保存: %s", output_path)


def parse_args():
    import argparse

    models = sorted(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(description="ETS Walk-forward 回测")
    parser.add_argument("--config", default="../../configs/ets/default.yaml", help="配置文件路径")
    parser.add_argument("--output", default="outputs/backtest.json", help="回测结果输出 JSON")
    parser.add_argument(
        "--data-profile",
        default="air_quality",
        help="数据集配置名，对应 configs/data/<profile>.yaml",
    )
    # 模型（与 train.py 对齐）
    parser.add_argument("--model", choices=models, default="ets_a")
    parser.add_argument("--hidden-size", default=128, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--bidirectional", default=False, action=argparse.BooleanOptionalAction)
    # 任务 / 数据
    parser.add_argument("--task", choices=["forecast", "classify"], default="forecast")
    parser.add_argument("--window-size", default=24, type=int)
    parser.add_argument("--horizon", default=1, type=int)
    parser.add_argument("--scale-target", default=True, action=argparse.BooleanOptionalAction)
    # 训练超参（每折重训用）
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", default=0, type=int)
    # 回测专用
    parser.add_argument("--n-splits", default=5, type=int, help="滚动切分折数")
    parser.add_argument(
        "--train-window",
        default=0,
        type=int,
        help="每折训练窗口长度，0 表示从序列起点开始",
    )
    parser.add_argument("--fold-epochs", default=5, type=int, help="每折最多训练轮数")
    return parser.parse_args()


if __name__ == "__main__":
    main()
