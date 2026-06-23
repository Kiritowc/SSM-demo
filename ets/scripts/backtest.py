#!/usr/bin/env python
"""Walk-forward backtest entry point."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo

REPO_ROOT = bootstrap_repo(_REPO)

from ets.engine.backtest import walk_forward_evaluate
from ets.models.registry import MODEL_REGISTRY
from ets.utils.config import load_config
from ets.utils.logger import setup_logger
from ets.utils.script_overrides import build_train_overrides


def main() -> None:
    args = parse_args()
    logger = setup_logger("ets")
    config_path = REPO_ROOT / args.config
    overrides = build_train_overrides(args)
    overrides.extend(
        [
            "eval.mode=walk_forward",
            f"eval.n_splits={args.n_splits}",
            f"eval.train_window={args.train_window or 'null'}",
            f"eval.fold_epochs={args.fold_epochs}",
        ]
    )
    cfg = load_config([config_path], overrides=overrides, project_root=REPO_ROOT)

    logger.info(
        "回测: 模型=%s | 任务=%s | fold_epochs=%d | n_splits=%d",
        cfg["model"]["name"],
        cfg["task"]["type"],
        args.fold_epochs,
        args.n_splits,
    )

    result = walk_forward_evaluate(cfg, project_root=REPO_ROOT)
    logger.info(
        "Walk-forward 完成: MAE=%.4f RMSE=%.4f (%d folds)",
        result["mae_mean"],
        result["rmse_mean"],
        len(result["folds"]),
    )

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("结果已保存: %s", output_path)


def parse_args():
    import argparse

    models = sorted(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="ets/configs/default.yaml")
    parser.add_argument("--output", default="outputs/backtest.json")
    parser.add_argument("--data-profile", default="air_quality")
    parser.add_argument("--model", choices=models, default="ets_b")
    parser.add_argument("--hidden-size", default=128, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--bidirectional", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--task", choices=["forecast", "classify"], default="forecast")
    parser.add_argument("--window-size", default=24, type=int)
    parser.add_argument("--horizon", default=1, type=int)
    parser.add_argument("--scale-target", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--train-window", default=0, type=int)
    parser.add_argument("--fold-epochs", default=5, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
