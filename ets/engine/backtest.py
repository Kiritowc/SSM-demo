"""Walk-forward and rolling evaluation for time series models."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from ets.data.ms_input import build_ms_inputs, get_input_mode, resolve_num_features
from ets.data.preprocess import load_and_preprocess
from ets.data.scaling import inverse_transform_targets
from ets.models.registry import build_model
from ets.tasks.factory import build_task
from ets.utils.metrics import compute_forecast_metrics
from ets.utils.seed import set_seed


def _make_windows(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    max_start = len(features) - window_size - horizon + 1
    xs, ys = [], []
    for start in range(0, max_start, stride):
        end = start + window_size
        xs.append(features[start:end])
        ys.append(targets[end : end + horizon])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

def walk_forward_evaluate(
    cfg: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Rolling walk-forward evaluation with per-fold retraining.
    Splits the timeline into ``eval.n_splits`` contiguous test blocks and
    trains a model on prior data for each block.
    """
    eval_cfg = cfg.get("eval", {})
    n_splits = int(eval_cfg.get("n_splits", 5))
    train_window = eval_cfg.get("train_window")
    stride = int(cfg["data"].get("stride", 1))
    root = Path(project_root) if project_root else Path.cwd()
    df = load_and_preprocess(cfg, project_root=str(root) if project_root else None)
    feature_cols = list(cfg["data"]["feature_cols"])
    target_col = cfg["data"]["target_col"]
    window_size = int(cfg["data"]["window_size"])
    horizon = int(cfg["data"]["horizon"])
    features = df[feature_cols].values.astype(np.float64)
    targets = df[target_col].values.astype(np.float64)
    total_len = len(df)
    split_size = max(total_len // n_splits, window_size + horizon + 1)
    fold_results: list[dict[str, float]] = []
    device = torch.device("cpu")
    set_seed(int(cfg.get("seed", 42)))
    for fold in range(n_splits):
        test_start = fold * split_size
        test_end = min((fold + 1) * split_size, total_len)
        if test_end - test_start < window_size + horizon:
            continue
        if train_window is not None:
            train_start = max(0, test_start - int(train_window))
        else:
            train_start = 0
        train_end = test_start
        if train_end - train_start < window_size + horizon + 1:
            continue
        train_features = features[train_start:train_end]
        train_targets = targets[train_start:train_end]
        test_features = features[test_start:test_end]
        test_targets = targets[test_start:test_end]
        feature_scaler = StandardScaler()
        feature_scaler.fit(train_features)
        train_features_s = feature_scaler.transform(train_features)
        test_features_s = feature_scaler.transform(test_features)
        target_scaler = None
        train_targets_s = train_targets
        test_targets_s = test_targets
        if bool(cfg["data"].get("scale_target", True)):
            target_scaler = StandardScaler()
            target_scaler.fit(train_targets.reshape(-1, 1))
            train_targets_s = target_scaler.transform(train_targets.reshape(-1, 1)).ravel()
            test_targets_s = target_scaler.transform(test_targets.reshape(-1, 1)).ravel()
        if get_input_mode(cfg) == "ms":
            train_features_s = build_ms_inputs(train_features_s, train_targets_s)
            test_features_s = build_ms_inputs(test_features_s, test_targets_s)
        x_train, y_train = _make_windows(
            train_features_s, train_targets_s, window_size, horizon, stride
        )
        x_test, y_test = _make_windows(
            test_features_s, test_targets_s, window_size, horizon, stride
        )
        if len(x_test) == 0:
            continue
        fold_cfg = deepcopy(cfg)
        fold_epochs = int(eval_cfg.get("fold_epochs", 5))
        fold_cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), fold_epochs)
        model = build_model(fold_cfg, resolve_num_features(fold_cfg, len(feature_cols)))
        model.to(device)
        task = build_task(fold_cfg, target_scaler=target_scaler)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
        batch_size = int(cfg["train"]["batch_size"])
        for _ in range(int(fold_cfg["train"]["epochs"])):
            model.train()
            for i in range(0, len(x_train), batch_size):
                batch_x = torch.from_numpy(x_train[i : i + batch_size]).to(device)
                batch_y = torch.from_numpy(y_train[i : i + batch_size]).to(device)
                optimizer.zero_grad()
                result = task.training_step(model, {"x": batch_x, "y": batch_y}, device)
                result["loss"].backward()
                optimizer.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x_test), batch_size):
                batch_x = torch.from_numpy(x_test[i : i + batch_size]).to(device)
                pred = model(batch_x).cpu().numpy()
                preds.append(pred)
        pred_np = np.concatenate(preds, axis=0)
        pred_np = inverse_transform_targets(target_scaler, pred_np)
        y_true = inverse_transform_targets(target_scaler, y_test)
        metrics = compute_forecast_metrics(y_true.reshape(-1, 1), pred_np.reshape(-1, 1))
        metrics["fold"] = float(fold)
        fold_results.append(metrics)
    if not fold_results:
        raise ValueError("Walk-forward produced no valid folds. Check data size and eval settings.")
    return {
        "n_splits": n_splits,
        "folds": fold_results,
        "mae_mean": float(np.mean([m["mae"] for m in fold_results])),
        "rmse_mean": float(np.mean([m["rmse"] for m in fold_results])),
    }
