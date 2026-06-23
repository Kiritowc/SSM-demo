"""TCN forecast models (NeuralForecast default + legacy stack for ets_t)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ets.models.base import BaseForecastModel
from ets.models.layers.mlp import MLP
from ets.models.layers.nf_tcn_encoder import TemporalConvolutionEncoder
from ets.models.layers.tcn import TemporalConvNet


class TCNModel(BaseForecastModel):
    """Bai-style stacked TCN + linear head (used by ``ets_t``)."""

    def __init__(
        self,
        input_size: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
        forecast_horizon: int,
        output_size: int = 1,
        seq_len: int = 96,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        self.seq_len = seq_len

        self.tcn = TemporalConvNet(
            input_size,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = nn.Linear(num_channels[-1], output_size * forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tcn(x.transpose(1, 2))
        hidden = y[:, :, -1]
        logits = self.head(hidden)
        if self.forecast_horizon == 1:
            return logits.view(-1, self.output_size)
        return logits.view(-1, self.forecast_horizon, self.output_size)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_features: int) -> TCNModel:
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        num_channels = model_cfg.get("num_channels", [32, 32, 32])
        if isinstance(num_channels, (list, tuple)):
            num_channels = [int(c) for c in num_channels]
        else:
            num_channels = [int(num_channels)]
        return cls(
            input_size=num_features,
            num_channels=num_channels,
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            forecast_horizon=int(data_cfg.get("horizon", 1)),
            seq_len=int(data_cfg["window_size"]),
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "variant": "stack_tcn",
            "input_size": self.input_size,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "task_type": self.task_type,
            "forecast_horizon": self.forecast_horizon,
            "output_size": self.output_size,
        }


class NfTCNModel(BaseForecastModel):
    """NeuralForecast-style TCN: dilated encoder + context adapter + MLP decoder."""

    def __init__(
        self,
        input_size: int,
        forecast_horizon: int,
        seq_len: int,
        kernel_size: int = 2,
        dilations: list[int] | None = None,
        encoder_hidden_size: int = 128,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.dilations = dilations or [1, 2, 4, 8, 16]
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.output_size = output_size

        self.hist_encoder = TemporalConvolutionEncoder(
            in_channels=input_size,
            out_channels=encoder_hidden_size,
            kernel_size=kernel_size,
            dilations=self.dilations,
        )
        self.context_adapter = nn.Linear(seq_len, forecast_horizon)
        self.mlp_decoder = MLP(
            in_features=encoder_hidden_size,
            out_features=output_size,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layers,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.hist_encoder(x)
        hidden = hidden.permute(0, 2, 1)
        context = self.context_adapter(hidden)
        context = context.permute(0, 2, 1)
        logits = self.mlp_decoder(context)
        if self.forecast_horizon == 1:
            return logits.view(-1, self.output_size)
        return logits.view(-1, self.forecast_horizon, self.output_size)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_features: int) -> NfTCNModel:
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        dilations = model_cfg.get("dilations", [1, 2, 4, 8, 16])
        if isinstance(dilations, (list, tuple)):
            dilations = [int(d) for d in dilations]
        else:
            dilations = [int(dilations)]
        seq_len = int(data_cfg["window_size"])
        horizon = int(data_cfg.get("horizon", 1))
        return cls(
            input_size=num_features,
            forecast_horizon=horizon,
            seq_len=seq_len,
            kernel_size=int(model_cfg.get("kernel_size", 2)),
            dilations=dilations,
            encoder_hidden_size=int(model_cfg.get("encoder_hidden_size", 128)),
            decoder_hidden_size=int(model_cfg.get("decoder_hidden_size", 128)),
            decoder_layers=int(model_cfg.get("decoder_layers", 2)),
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "variant": "nf_tcn",
            "input_size": self.input_size,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "encoder_hidden_size": self.encoder_hidden_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "decoder_layers": self.decoder_layers,
            "task_type": self.task_type,
            "forecast_horizon": self.forecast_horizon,
            "output_size": self.output_size,
        }
