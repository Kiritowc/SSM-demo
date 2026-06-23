"""MLP decoder (NeuralForecast-compatible)."""

from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    """Feed-forward decoder with shared hidden width."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        if num_layers == 1:
            self.layers = nn.Sequential(nn.Linear(in_features, out_features))
            return

        act = getattr(nn, activation)()
        layers: list[nn.Module] = [
            nn.Linear(in_features, hidden_size),
            act,
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                act,
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_size, out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
