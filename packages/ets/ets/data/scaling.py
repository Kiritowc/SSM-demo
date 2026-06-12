"""Target scaling helpers for forecast tasks."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def inverse_transform_targets(
    scaler: StandardScaler | None,
    values: np.ndarray,
) -> np.ndarray:
    """Restore scaled targets/predictions to the original value scale."""
    if scaler is None:
        return values

    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    restored = scaler.inverse_transform(flat)
    return restored.reshape(original_shape)
