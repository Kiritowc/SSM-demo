"""Trend-only linear branch with shared decomposition for seasonal routing."""

from __future__ import annotations

import torch
import torch.nn as nn

from ets.models.layers.decomposition import SeriesDecomp, resolve_decomp_kernel


class TrendOnlyLinear(nn.Module):
    """Series decomposition: trend -> Linear forecast; seasonal kept for TCN input."""

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
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decomposition(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, trend = self.decomposition(x)
        return self.linear_trend(trend.squeeze(-1))
