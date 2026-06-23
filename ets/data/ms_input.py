"""Multivariate-input / single-target (MS) input helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_input_mode(cfg: dict[str, Any]) -> str:
    """Return ``features_only`` or ``ms``."""
    return str(cfg.get("data", {}).get("input_mode", "features_only"))


def resolve_num_features(cfg: dict[str, Any], num_feature_cols: int) -> int:
    """Model input width after applying the configured input mode."""
    if get_input_mode(cfg) == "ms":
        return num_feature_cols + 1
    return num_feature_cols


def build_ms_inputs(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Concatenate scaled feature matrix with target history.

    Args:
        features: (T, num_feature_cols)
        targets: (T,)

    Returns:
        (T, num_feature_cols + 1)
    """
    target_col = targets.reshape(-1, 1).astype(features.dtype, copy=False)
    return np.concatenate([features, target_col], axis=1)
