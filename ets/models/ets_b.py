"""EtsB: full-channel decomposition-linear + last-step gated TCN residual."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ets.models.base import BaseForecastModel
from ets.models.layers.decomp_linear import FullChannelDecompLinear
from ets.models.layers.tcn import TemporalConvNet


class EtsBModel(BaseForecastModel):
    """Forecaster: FullChannelDecompLinear + last-step gated TCN residual."""

    def __init__(
        self,
        input_size: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
        forecast_horizon: int,
        seq_len: int,
        decomp_kernel: int = 9,
        use_weight_norm: bool = False,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.forecast_horizon = forecast_horizon
        self.seq_len = seq_len
        self.decomp_kernel = decomp_kernel
        self.use_weight_norm = use_weight_norm
        self.output_size = output_size
        self.hidden_dim = num_channels[-1]

        c_out = output_size * forecast_horizon
        self.decomp_linear = FullChannelDecompLinear(
            seq_len=seq_len,
            pred_len=forecast_horizon,
            channels=input_size,
            decomp_kernel=decomp_kernel,
        )
        self.tcn = TemporalConvNet(
            input_size,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
        )
        self.tcn_head = nn.Linear(self.hidden_dim, c_out)
        self.gate = nn.Linear(input_size, c_out)

    def _tcn_logits(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.tcn(x.transpose(1, 2))
        return self.tcn_head(hidden[:, :, -1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_lin = self.decomp_linear(x)
        y_tcn = self._tcn_logits(x)
        gate = torch.sigmoid(self.gate(x[:, -1, :]))
        logits = y_lin + gate * y_tcn
        if self.forecast_horizon == 1:
            return logits.view(-1, self.output_size)
        return logits.view(-1, self.forecast_horizon, self.output_size)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_features: int) -> EtsBModel:
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        num_channels = model_cfg.get("num_channels", [24, 24])
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
            decomp_kernel=int(model_cfg.get("decomp_kernel", 9)),
            use_weight_norm=bool(model_cfg.get("use_weight_norm", False)),
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "variant": "gated_last_step_scale",
            "input_size": self.input_size,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "decomp_kernel": self.decomp_kernel,
            "use_weight_norm": self.use_weight_norm,
            "task_type": self.task_type,
            "forecast_horizon": self.forecast_horizon,
            "output_size": self.output_size,
            "num_params": sum(p.numel() for p in self.parameters()),
        }
