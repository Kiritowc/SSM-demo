"""NeuralForecast-style dilated causal TCN encoder."""

from __future__ import annotations

import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.chomp = Chomp1d(padding)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        return self.activation(self.chomp(self.conv(x)))


class TemporalConvolutionEncoder(nn.Module):
    """Stack of dilated causal convolutions; I/O layout (N, T, C)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilations: list[int],
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for dilation in dilations:
            layers.append(
                CausalConv1d(
                    in_channels=channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation,
                    activation=activation,
                )
            )
            channels = out_channels
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.tcn(x)
        return x.permute(0, 2, 1).contiguous()
