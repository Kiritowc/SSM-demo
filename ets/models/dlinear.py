"""DLinear forecast model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ets.models.base import BaseForecastModel
from ets.models.layers.decomposition import SeriesDecomp, resolve_decomp_kernel


class _DLinearCore(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        individual: bool = False,
        decomp_kernel: int = 25,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.pred_len = pred_len
        self.individual = individual
        kernel_size = resolve_decomp_kernel(decomp_kernel, seq_len)
        self.decomposition = SeriesDecomp(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(channels)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(channels)]
            )
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                seasonal_init.size(0),
                seasonal_init.size(1),
                self.pred_len,
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                trend_init.size(0),
                trend_init.size(1),
                self.pred_len,
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        return (seasonal_output + trend_output).permute(0, 2, 1)


class DLinearModel(BaseForecastModel):
    """Decomposition-linear model with readout for MS forecasting."""

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        pred_len: int,
        individual: bool = False,
        decomp_kernel: int = 25,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.decomp_kernel = decomp_kernel
        self.output_size = output_size
        self.forecast_horizon = pred_len

        self.core = _DLinearCore(
            seq_len=seq_len,
            pred_len=pred_len,
            channels=input_size,
            individual=individual,
            decomp_kernel=decomp_kernel,
        )
        self.readout = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.core(x)
        logits = self.readout(out).squeeze(-1)
        if self.pred_len == 1:
            return logits.view(-1, self.output_size)
        return logits.unsqueeze(-1)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_features: int) -> DLinearModel:
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        return cls(
            input_size=num_features,
            seq_len=int(data_cfg["window_size"]),
            pred_len=int(data_cfg.get("horizon", 1)),
            individual=bool(model_cfg.get("individual", False)),
            decomp_kernel=int(model_cfg.get("decomp_kernel", 25)),
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "individual": self.individual,
            "decomp_kernel": self.decomp_kernel,
            "task_type": self.task_type,
            "forecast_horizon": self.forecast_horizon,
            "output_size": self.output_size,
        }
