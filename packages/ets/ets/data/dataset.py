"""Sliding window dataset for time series modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Create sliding windows from preprocessed time series data.

    For forecast: X shape (window_size, num_features), y shape (horizon,) or (horizon, output_size)
    For classify: X shape (window_size, num_features), y shape () class index at window end
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int,
        horizon: int,
        task_type: str = "forecast",
        stride: int = 1,
    ) -> None:
        self.features = features.astype(np.float32)
        self.targets = targets
        self.window_size = window_size
        self.horizon = horizon
        self.task_type = task_type
        self.stride = stride

        if task_type == "forecast":
            max_start = len(features) - window_size - horizon + 1
        else:
            max_start = len(features) - window_size + 1

        if max_start <= 0:
            raise ValueError(
                f"Not enough samples: len={len(features)}, window_size={window_size}, "
                f"horizon={horizon}, task_type={task_type}"
            )

        self.indices = list(range(0, max_start, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = self.indices[idx]
        end = start + self.window_size

        x = self.features[start:end]

        if self.task_type == "forecast":
            y_start = end
            y_end = y_start + self.horizon
            y = self.targets[y_start:y_end]
            if y.ndim == 1:
                y = y.astype(np.float32)
            else:
                y = y.astype(np.float32)
        else:
            label_idx = end - 1
            y = int(self.targets[label_idx])

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y) if self.task_type == "classify" else torch.from_numpy(
                np.asarray(y, dtype=np.float32)
            ),
        }
