"""Base sequence model with shared task heads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from ets.models.layers.mlp import MLP


class BaseForecastModel(nn.Module, ABC):
    """Base class for non-RNN forecast models (TCN, DLinear, etc.)."""

    task_type: str = "forecast"

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict[str, Any], num_features: int) -> BaseForecastModel:
        """Build model from merged config."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for logging and export."""


class BaseSequenceModel(nn.Module, ABC):
    """
    RNN encoder + task head.

    Forecast uses NeuralForecast-style direct multi-step decoding:
    encoder sequence output -> last ``horizon`` steps (or upsample) -> MLP decoder.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        task_type: str,
        output_size: int = 1,
        forecast_horizon: int = 1,
        num_classes: int = 2,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        seq_len: int = 96,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.task_type = task_type
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon
        self.num_classes = num_classes
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.seq_len = seq_len

        self.num_directions = 2 if bidirectional else 1
        self.encoder_output_size = hidden_size * self.num_directions

        self.rnn = self._build_rnn()
        self.dropout_layer = nn.Dropout(dropout)

        if task_type == "forecast":
            self.mlp_decoder = MLP(
                in_features=self.encoder_output_size,
                out_features=output_size,
                hidden_size=decoder_hidden_size,
                num_layers=decoder_layers,
                dropout=0.0,
            )
            if forecast_horizon > seq_len:
                self.upsample_sequence = nn.Linear(seq_len, forecast_horizon)
            else:
                self.upsample_sequence = None
            self.head = None
        elif task_type == "classify":
            self.head = nn.Linear(self.encoder_output_size, num_classes)
            self.mlp_decoder = None
            self.upsample_sequence = None
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    @abstractmethod
    def _build_rnn(self) -> nn.Module:
        """Build the RNN encoder (LSTM or GRU)."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Last-step hidden for classification."""
        output, _ = self.rnn(x)
        if self.bidirectional:
            forward = output[:, -1, : self.hidden_size]
            backward = output[:, 0, self.hidden_size :]
            hidden = torch.cat([forward, backward], dim=-1)
        else:
            hidden = output[:, -1, :]
        return self.dropout_layer(hidden)

    def _forecast_hidden(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        output = self.dropout_layer(output)
        h = self.forecast_horizon
        if self.upsample_sequence is not None:
            hidden = output.permute(0, 2, 1)
            hidden = self.upsample_sequence(hidden)
            return hidden.permute(0, 2, 1)
        return output[:, -h:, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classify":
            return self.head(self.encode(x))

        hidden = self._forecast_hidden(x)
        logits = self.mlp_decoder(hidden)
        if self.forecast_horizon == 1:
            return logits.view(-1, self.output_size)
        return logits.view(-1, self.forecast_horizon, self.output_size)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_features: int) -> BaseSequenceModel:
        """Build model from merged config."""
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        task_type = cfg["task"]["type"]
        forecast_horizon = int(data_cfg.get("horizon", 1))

        kwargs = {
            "input_size": num_features,
            "hidden_size": int(model_cfg["hidden_size"]),
            "num_layers": int(model_cfg["num_layers"]),
            "dropout": float(model_cfg.get("dropout", 0.0)),
            "bidirectional": bool(model_cfg.get("bidirectional", False)),
            "task_type": task_type,
            "output_size": 1,
            "forecast_horizon": forecast_horizon,
            "decoder_hidden_size": int(model_cfg.get("decoder_hidden_size", 128)),
            "decoder_layers": int(model_cfg.get("decoder_layers", 2)),
            "seq_len": int(data_cfg["window_size"]),
        }

        if task_type == "classify":
            kwargs["num_classes"] = int(data_cfg["classify"]["num_classes"])

        return cls(**kwargs)

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for logging and export."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "decoder_hidden_size": self.decoder_hidden_size,
            "decoder_layers": self.decoder_layers,
            "task_type": self.task_type,
            "forecast_horizon": self.forecast_horizon,
            "output_size": self.output_size,
            "num_classes": self.num_classes if self.task_type == "classify" else None,
        }
