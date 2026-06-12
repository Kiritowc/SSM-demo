"""Lightweight inference bundle without full data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ets.data.ms_input import get_input_mode, resolve_num_features
from ets.data.scaling import inverse_transform_targets
from ets.models.registry import build_model
from ets.utils.checkpoint import load_checkpoint, load_scaler, load_target_scaler
from ets.utils.device import get_device


class InferenceBundle:
    """Load checkpoint, model, and scalers for inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        cfg: dict[str, Any] | None = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = get_device(device)

        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.cfg = cfg or state["cfg"]
        self.epoch = state.get("epoch", 0)
        self.metrics = state.get("metrics", {})
        self.task_type = self.cfg["task"]["type"]

        num_features = resolve_num_features(self.cfg, len(self.cfg["data"]["feature_cols"]))
        self.model = build_model(self.cfg, num_features)
        load_checkpoint(self.checkpoint_path, self.model, device=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.feature_scaler = load_scaler(self.checkpoint_path)
        self.target_scaler = load_target_scaler(self.checkpoint_path)
        if self.feature_scaler is None:
            raise FileNotFoundError(
                f"Feature scaler not found alongside checkpoint: {self.checkpoint_path}"
            )

    def predict_window(self, window: np.ndarray) -> np.ndarray:
        """Predict from window array of shape (window_size, num_features) or (1, W, F)."""
        if window.ndim == 2:
            window = window[np.newaxis, ...]

        scaled = self._scale_input_window(window)
        x = torch.from_numpy(scaled.astype(np.float32)).to(self.device)

        with torch.no_grad():
            pred = self.model(x)

        if self.task_type == "classify":
            return pred.argmax(dim=-1).cpu().numpy()

        pred_np = pred.cpu().numpy()
        return inverse_transform_targets(self.target_scaler, pred_np)

    def _scale_input_window(self, window: np.ndarray) -> np.ndarray:
        if get_input_mode(self.cfg) == "ms" and self.task_type == "forecast":
            n_feat = len(self.cfg["data"]["feature_cols"])
            expected = n_feat + 1
            if window.shape[-1] != expected:
                raise ValueError(
                    f"MS mode expects input width {expected} "
                    f"(features + target history), got {window.shape[-1]}"
                )
            features = window[..., :n_feat]
            target_hist = window[..., n_feat:]
            scaled_features = self.feature_scaler.transform(
                features.reshape(-1, n_feat)
            ).reshape(features.shape)
            if self.target_scaler is not None:
                scaled_target = self.target_scaler.transform(
                    target_hist.reshape(-1, 1)
                ).reshape(target_hist.shape)
            else:
                scaled_target = target_hist
            return np.concatenate([scaled_features, scaled_target], axis=-1)

        scaled = self.feature_scaler.transform(window.reshape(-1, window.shape[-1]))
        return scaled.reshape(window.shape)
