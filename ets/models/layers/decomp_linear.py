"""Full-channel decomposition-linear branch with readout."""

from __future__ import annotations

import torch
import torch.nn as nn

from ets.models.layers.decomposition import SeriesDecomp, resolve_decomp_kernel


class FullChannelDecompLinear(nn.Module):
    """Series decomposition + shared dual linear on all channels + readout."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        decomp_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        kernel_size = resolve_decomp_kernel(decomp_kernel, seq_len)
        self.decomposition = SeriesDecomp(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.readout = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        seasonal_out = self.linear_seasonal(seasonal)
        trend_out = self.linear_trend(trend)
        combined = (seasonal_out + trend_out).permute(0, 2, 1)
        return self.readout(combined).squeeze(-1)
