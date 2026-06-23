#!/usr/bin/env python
"""Training entry point."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo, ensure_runtime_python

REPO_ROOT = bootstrap_repo(_REPO)
ensure_runtime_python("numpy", "torch")
from ets.data.datamodule import DataModule
from ets.engine.evaluator import Evaluator
from ets.engine.trainer import Trainer
from ets.models.registry import MODEL_REGISTRY, build_model
from ets.utils.checkpoint import (
    load_scaler,
    load_target_scaler,
    resolve_checkpoint,
    run_dir_from_checkpoint,
)
from ets.utils.config import load_config
from ets.utils.device import get_device
from ets.utils.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    wrap_model_ddp,
)
from ets.utils.logger import log_plain, setup_logger
from ets.utils.script_overrides import build_train_overrides, resolve_early_stopping
from ets.utils.seed import set_seed
from ets.utils.visualizer import plot_predictions


def main() -> None:
    args = parse_args()
    requested_monitor = args.monitor
    args.monitor, args.monitor_mode = resolve_early_stopping(args.task, args.monitor)
    config_path = REPO_ROOT / args.config
    cfg = load_config(
        [config_path],
        overrides=build_train_overrides(args),
        project_root=REPO_ROOT,
    )
    local_rank = setup_distributed(args.local_rank)
    logger = setup_logger("ets", style="train")
    if is_main_process():
        logger.info("配置: %s", config_path)
        if requested_monitor != args.monitor:
            logger.info(
                "任务=%s 已将 monitor 从 %s 调整为 %s (mode=%s)",
                args.task,
                requested_monitor,
                args.monitor,
                args.monitor_mode,
            )
        logger.info(
            "模型=%s | 任务=%s | epochs=%d | batch=%d | lr=%s",
            cfg["model"]["name"],
            cfg["task"]["type"],
            cfg["train"]["epochs"],
            cfg["train"]["batch_size"],
            cfg["train"]["lr"],
        )
    else:
        logger.disabled = True
    set_seed(int(cfg.get("seed", 42)))
    train_cfg = cfg["train"]
    resume_path = train_cfg.get("resume")
    if resume_path:
        resume_candidate = Path(resume_path)
        if not resume_candidate.is_absolute():
            resume_candidate = REPO_ROOT / resume_candidate
        resume_file = resolve_checkpoint(resume_candidate, prefer="last")
        resume_path = str(resume_file)
        cfg["train"]["resume"] = resume_path
        run_dir = run_dir_from_checkpoint(resume_file)
    else:
        run_prefix = train_cfg.get("exp_name") or str(cfg["model"]["name"]).lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(train_cfg["log_dir"])
        if not log_dir.is_absolute():
            log_dir = REPO_ROOT / log_dir
        run_dir = log_dir / f"{run_prefix}_{timestamp}"
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.get("device", "auto"))
    if local_rank >= 0 and device.type == "cuda":
        device = get_device("cuda")
    data_module = DataModule(cfg, project_root=str(REPO_ROOT))
    bundle = data_module.setup()
    if is_main_process():
        logger.info(
            "数据集就绪: 训练集=%d, 验证集=%d, 测试集=%d, 特征数=%d",
            len(bundle.train_loader.dataset),
            len(bundle.val_loader.dataset),
            len(bundle.test_loader.dataset),
            bundle.num_features,
        )
    model = build_model(cfg, bundle.num_features)
    model.to(device)
    model = wrap_model_ddp(model, local_rank)
    feature_scaler = bundle.feature_scaler
    target_scaler = bundle.target_scaler
    if resume_path:
        feature_scaler = load_scaler(resume_path) or feature_scaler
        target_scaler = load_target_scaler(resume_path) or target_scaler
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        device=device,
        run_dir=run_dir,
        scaler=feature_scaler,
        target_scaler=target_scaler,
        logger=logger,
    )
    result = trainer.fit()
    if is_main_process():
        raw_model = model.module if hasattr(model, "module") else model
        evaluator = Evaluator(
            raw_model,
            cfg,
            device,
            target_scaler=bundle.target_scaler,
            logger=logger,
        )
        evaluator.evaluate(bundle.test_loader, split_name="测试集")
        vis_cfg = cfg.get("train", {}).get("visualization", {})
        if vis_cfg.get("enabled", True) and vis_cfg.get("plot_predictions", True):
            plot_dir = run_dir / vis_cfg.get("save_dir", "plots")
            preds = evaluator.predict(bundle.test_loader)
            plot_predictions(
                preds["targets"],
                preds["predictions"],
                plot_dir,
                task_type=cfg["task"]["type"],
                max_samples=int(vis_cfg.get("max_prediction_samples", 500)),
                dpi=int(vis_cfg.get("dpi", 150)),
                logger=logger,
            )
        log_plain(
            logger,
            f"训练完成, Best {result.get('monitor', 'val_loss')}: {result['best_monitor_score']:.6f}",
        )
        log_plain(logger, f"输出目录: {run_dir}")
    cleanup_distributed()

def parse_args():
    import argparse
    from ets.utils.script_overrides import CLASSIFY_MONITORS, FORECAST_MONITORS

    if not hasattr(argparse, "BooleanOptionalAction"):
        class BooleanOptionalAction(argparse.Action):
            def __init__(self, option_strings, dest, default=None, **kwargs):
                if default is None:
                    default = False
                opts: list[str] = []
                for opt in option_strings:
                    if opt.startswith("--no-"):
                        continue
                    opts.append(opt)
                    opts.append(f"--no-{opt[2:]}")
                super().__init__(opts, dest, nargs=0, default=default, **kwargs)

            def __call__(self, parser, namespace, values, option_string=None):
                if option_string is not None and option_string.startswith("--no-"):
                    setattr(namespace, self.dest, False)
                else:
                    setattr(namespace, self.dest, True)

        argparse.BooleanOptionalAction = BooleanOptionalAction

    models = sorted(MODEL_REGISTRY.keys())
    monitors = sorted(set(FORECAST_MONITORS + CLASSIFY_MONITORS))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="ets/configs/default.yaml")
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
    parser.add_argument("--input-mode", default=None, choices=["ms", "features_only"])
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--scheduler", default="")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--log-interval", default=10, type=int)
    parser.add_argument("--early-stopping", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--monitor", default="val_rmse", choices=monitors)
    parser.add_argument("--resume", default="")
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--num-channels", default="")
    parser.add_argument("--dilations", default="")
    parser.add_argument("--encoder-hidden-size", default=None, type=int)
    parser.add_argument("--decoder-hidden-size", default=None, type=int)
    parser.add_argument("--decoder-layers", default=None, type=int)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--kernel-size", default=None, type=int)
    parser.add_argument("--no-visualization", default=False, action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    main()
