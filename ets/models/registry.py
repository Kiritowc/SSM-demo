"""Model registry for building models by name."""

from __future__ import annotations

from typing import Any, Type, Union

import torch.nn as nn

from ets.models.base import BaseForecastModel, BaseSequenceModel
from ets.models.dlinear import DLinearModel
from ets.models.gru import GRUModel
from ets.models.lstm import LSTMModel
from ets.models.ets_a import EtsAModel
from ets.models.ets_b import EtsBModel
from ets.models.ets_c import EtsCModel
from ets.models.ets_h import EtsHModel
from ets.models.tcn import NfTCNModel, TCNModel

ModelType = Union[Type[BaseSequenceModel], Type[BaseForecastModel]]

MODEL_REGISTRY: dict[str, ModelType] = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "tcn": NfTCNModel,
    "ets_t": TCNModel,
    "dlinear": DLinearModel,
    "ets_a": EtsAModel,
    "ets_m": EtsAModel,
    "ets_b": EtsBModel,
    "ets_c": EtsCModel,
    "ets_h": EtsHModel,
}


def register_model(name: str, model_cls: ModelType) -> None:
    """Register a new model class."""
    MODEL_REGISTRY[name.lower()] = model_cls


def build_model(cfg: dict[str, Any], num_features: int) -> nn.Module:
    """Build model from config using registry."""
    name = cfg["model"]["name"].lower()
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    model_cls = MODEL_REGISTRY[name]
    return model_cls.from_config(cfg, num_features)
