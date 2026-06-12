"""DataModule for loading, scaling, and batching time series data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, DistributedSampler

from ets.data.dataset import SlidingWindowDataset
from ets.data.ms_input import build_ms_inputs, get_input_mode, resolve_num_features
from ets.data.preprocess import load_and_preprocess, temporal_split
from ets.utils.distributed import is_distributed


@dataclass
class DataBundle:
    """Container for datasets and metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_scaler: StandardScaler
    target_scaler: StandardScaler | None
    num_features: int
    output_size: int
    num_classes: int
    target_col: str | None

    @property
    def scaler(self) -> StandardScaler:
        """Backward-compatible alias for the feature scaler."""
        return self.feature_scaler


class DataModule:
    """Prepare train/val/test dataloaders with temporal split and scaling."""

    def __init__(self, cfg: dict[str, Any], project_root: str | None = None) -> None:
        self.cfg = cfg
        self.project_root = project_root
        self.data_cfg = cfg["data"]
        self.task_type = cfg["task"]["type"]
        self.feature_scaler = StandardScaler()
        self.target_scaler: StandardScaler | None = None
        self._bundle: DataBundle | None = None

    def setup(self) -> DataBundle:
        """Load data, split, scale, and create dataloaders."""
        df = load_and_preprocess(self.cfg, project_root=self.project_root)

        feature_cols = list(self.data_cfg["feature_cols"])
        target_col = self.data_cfg.get("target_col")
        window_size = int(self.data_cfg["window_size"])
        horizon = int(self.data_cfg["horizon"])
        stride = int(self.data_cfg.get("stride", 1))
        train_ratio = float(self.data_cfg.get("train_ratio", 0.7))
        val_ratio = float(self.data_cfg.get("val_ratio", 0.15))

        train_df, val_df, test_df = temporal_split(df, train_ratio, val_ratio)

        train_features = train_df[feature_cols].values
        self.feature_scaler.fit(train_features)

        train_features = self.feature_scaler.transform(train_df[feature_cols].values)
        val_features = self.feature_scaler.transform(val_df[feature_cols].values)
        test_features = self.feature_scaler.transform(test_df[feature_cols].values)

        if self.task_type == "classify":
            train_targets = train_df["label"].values
            val_targets = val_df["label"].values
            test_targets = test_df["label"].values
            output_size = 1
            num_classes = int(self.data_cfg["classify"]["num_classes"])
            self.target_scaler = None
        else:
            if target_col is None:
                raise ValueError("target_col is required for forecast task.")
            train_targets = train_df[target_col].values.astype(np.float64)
            val_targets = val_df[target_col].values.astype(np.float64)
            test_targets = test_df[target_col].values.astype(np.float64)
            output_size = 1
            num_classes = 0

            if bool(self.data_cfg.get("scale_target", True)):
                self.target_scaler = StandardScaler()
                self.target_scaler.fit(train_targets.reshape(-1, 1))
                train_targets = self.target_scaler.transform(
                    train_targets.reshape(-1, 1)
                ).ravel()
                val_targets = self.target_scaler.transform(
                    val_targets.reshape(-1, 1)
                ).ravel()
                test_targets = self.target_scaler.transform(
                    test_targets.reshape(-1, 1)
                ).ravel()
            else:
                self.target_scaler = None

        if self.task_type == "forecast" and get_input_mode(self.cfg) == "ms":
            train_features = build_ms_inputs(train_features, train_targets)
            val_features = build_ms_inputs(val_features, val_targets)
            test_features = build_ms_inputs(test_features, test_targets)

        train_ds = SlidingWindowDataset(
            train_features, train_targets, window_size, horizon, self.task_type, stride
        )
        val_ds = SlidingWindowDataset(
            val_features, val_targets, window_size, horizon, self.task_type, stride
        )
        test_ds = SlidingWindowDataset(
            test_features, test_targets, window_size, horizon, self.task_type, stride
        )

        train_cfg = self.cfg["train"]
        batch_size = int(train_cfg["batch_size"])
        num_workers = int(train_cfg.get("num_workers", 0))
        pin_memory = bool(train_cfg.get("pin_memory", False))

        train_sampler = None
        shuffle = True
        if is_distributed():
            train_sampler = DistributedSampler(train_ds, shuffle=True)
            shuffle = False

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self._bundle = DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            feature_scaler=self.feature_scaler,
            target_scaler=self.target_scaler,
            num_features=resolve_num_features(self.cfg, len(feature_cols)),
            output_size=output_size,
            num_classes=num_classes,
            target_col=target_col,
        )

        return self._bundle

    @property
    def bundle(self) -> DataBundle:
        if self._bundle is None:
            raise RuntimeError("Call setup() before accessing bundle.")
        return self._bundle
