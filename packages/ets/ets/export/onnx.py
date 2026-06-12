"""ONNX export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ets.data.ms_input import resolve_num_features
from ets.models.registry import build_model
from ets.utils.checkpoint import load_checkpoint
from ets.utils.device import get_device
from ets.utils.logger import setup_logger


def _validate_onnx(output_path: Path, dummy_input: torch.Tensor, torch_output: np.ndarray) -> None:
    """Run onnx checker and ORT smoke test against PyTorch output."""
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_input = dummy_input.detach().cpu().numpy().astype(np.float32)
    ort_output = session.run(None, {"input": ort_input})[0]

    if not np.allclose(ort_output, torch_output, rtol=1e-4, atol=1e-4):
        raise RuntimeError(
            "ONNX Runtime output differs from PyTorch. "
            f"max_diff={float(np.max(np.abs(ort_output - torch_output)))}"
        )


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    cfg: dict[str, Any] | None = None,
    opset_version: int = 17,
    device: str = "cpu",
) -> Path:
    """Export trained model to ONNX format with validation metadata."""
    logger = setup_logger("ets.export")
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dev = get_device(device)
    state = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    resolved_cfg = cfg or state["cfg"]

    window_size = int(resolved_cfg["data"]["window_size"])
    num_features = resolve_num_features(resolved_cfg, len(resolved_cfg["data"]["feature_cols"]))
    scale_target = bool(resolved_cfg["data"].get("scale_target", True))

    model = build_model(resolved_cfg, num_features)
    load_checkpoint(checkpoint_path, model, device=dev)
    model.eval()

    dummy_input = torch.randn(1, window_size, num_features, device=dev)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    with torch.no_grad():
        torch_output = model(dummy_input).detach().cpu().numpy()

    _validate_onnx(output_path, dummy_input, torch_output)

    feature_scaler_path = checkpoint_path.parent / "scaler.joblib"
    target_scaler_path = checkpoint_path.parent / "target_scaler.joblib"
    meta = {
        "checkpoint": str(checkpoint_path.resolve()),
        "onnx_path": str(output_path.resolve()),
        "opset_version": opset_version,
        "input_shape": [1, window_size, num_features],
        "input_names": ["input"],
        "output_names": ["output"],
        "output_scaled": True,
        "scale_target": scale_target,
        "feature_scaler": (
            str(feature_scaler_path.resolve()) if feature_scaler_path.exists() else None
        ),
        "target_scaler": (
            str(target_scaler_path.resolve()) if target_scaler_path.exists() else None
        ),
        "task_type": resolved_cfg["task"]["type"],
        "note": (
            "ONNX output is in scaled model space; "
            "apply target_scaler.inverse_transform for forecast."
        ),
    }
    meta_path = output_path.parent / "export_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("ONNX model exported to %s (opset=%d)", output_path, opset_version)
    logger.info("Export metadata saved to %s", meta_path)
    return output_path

