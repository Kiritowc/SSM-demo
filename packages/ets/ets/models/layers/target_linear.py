"""Target-only decomposition-linear branch for MS forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from ets.models.layers.decomposition import SeriesDecomp, resolve_decomp_kernel


class TargetDecompLinear(nn.Module):
    """Series decomposition + dual linear on a single target series (B, T, 1)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        decomp_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        kernel_size = resolve_decomp_kernel(decomp_kernel, seq_len)
        self.decomposition = SeriesDecomp(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.squeeze(-1)
        trend = trend.squeeze(-1)
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)
