"""EtsA: DLinear-style full-channel decomposition + dual-branch TCN (features_only)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ets.models.base import BaseForecastModel
from ets.models.layers.decomposition import SeriesDecomp, resolve_decomp_kernel
from ets.models.layers.tcn import TemporalConvNet


def resolve_target_feature_index(
    cfg: dict[str, Any],
    feature_cols: list[str],
) -> int:
    """Index of the supervised target within ``feature_cols`` (features_only)."""
    idx = int(cfg.get("data", {}).get("target_channel", -1))
    if idx >= 0:
        return idx
    target_col = str(cfg["data"]["target_col"])
    if target_col not in feature_cols:
        raise ValueError(
            f"target_col '{target_col}' must appear in data.feature_cols for ets_m (features_only)"
        )
    return feature_cols.index(target_col)


def _format_output(
    logits: torch.Tensor,
    forecast_horizon: int,
    output_size: int,
) -> torch.Tensor:
    if forecast_horizon == 1:
        return logits.view(-1, output_size)
    return logits.view(-1, forecast_horizon, output_size)


class EtsAModel(BaseForecastModel):
    """Full-channel SeriesDecomp + seasonal/trend dual TCN branches."""

    def __init__(
        self,
        input_size: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
        forecast_horizon: int,
        seq_len: int,
        individual: bool = False,
        target_feature_idx: int | None = None,
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
        self.individual = individual
        self.target_feature_idx = target_feature_idx
        self.decomp_kernel = decomp_kernel
        self.use_weight_norm = use_weight_norm
        self.output_size = output_size
        self.hidden_dim = num_channels[-1]
        c_out = output_size * forecast_horizon

        decomp_k = resolve_decomp_kernel(decomp_kernel, seq_len)
        self.decomposition = SeriesDecomp(decomp_k)

        if individual:
            if target_feature_idx is None:
                raise ValueError("target_feature_idx is required when individual=True")
            self.seasonal_tcns = nn.ModuleList(
                [
                    TemporalConvNet(
                        1,
                        num_channels,
                        kernel_size=kernel_size,
                        dropout=dropout,
                        use_weight_norm=use_weight_norm,
                    )
                    for _ in range(input_size)
                ]
            )
            self.trend_tcns = nn.ModuleList(
                [
                    TemporalConvNet(
                        1,
                        num_channels,
                        kernel_size=kernel_size,
                        dropout=dropout,
                        use_weight_norm=use_weight_norm,
                    )
                    for _ in range(input_size)
                ]
            )
            self.seasonal_heads = nn.ModuleList(
                [nn.Linear(self.hidden_dim, c_out) for _ in range(input_size)]
            )
            self.trend_heads = nn.ModuleList(
                [nn.Linear(self.hidden_dim, c_out) for _ in range(input_size)]
            )
        else:
            self.seasonal_tcn = TemporalConvNet(
                input_size,
                num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                use_weight_norm=use_weight_norm,
            )
            self.trend_tcn = TemporalConvNet(
                input_size,
                num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                use_weight_norm=use_weight_norm,
            )
            self.seasonal_head = nn.Linear(self.hidden_dim, c_out)
            self.trend_head = nn.Linear(self.hidden_dim, c_out)

    def _shared_branch(
        self,
        x: torch.Tensor,
        tcn: TemporalConvNet,
        head: nn.Linear,
    ) -> torch.Tensor:
        hidden = tcn(x.transpose(1, 2))[:, :, -1]
        return head(hidden)

    def _individual_branch(
        self,
        x: torch.Tensor,
        tcns: nn.ModuleList,
        heads: nn.ModuleList,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        outputs = torch.zeros(
            batch_size,
            self.input_size,
            self.output_size * self.forecast_horizon,
            dtype=x.dtype,
            device=x.device,
        )
        for i in range(self.input_size):
            channel = x[:, :, i].unsqueeze(1)
            hidden = tcns[i](channel)[:, :, -1]
            outputs[:, i, :] = heads[i](hidden)
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomposition(x)

        if self.individual:
            y_s = self._individual_branch(seasonal, self.seasonal_tcns, self.seasonal_heads)
            y_t = self._individual_branch(trend, self.trend_tcns, self.trend_heads)
            logits = y_s + y_t
            assert self.target_feature_idx is not None
            logits = logits[:, self.target_feature_idx, :]
        else:
            y_s = self._shared_branch(seasonal, self.seasonal_tcn, self.seasonal_head)
            y_t = self._shared_branch(trend, self.trend_tcn, self.trend_head)
            logits = y_s + y_t

        return _format_output(logits, self.forecast_horizon, self.output_size)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_features: int) -> EtsAModel:
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        num_channels = model_cfg.get("num_channels", [16, 16])
        if isinstance(num_channels, (list, tuple)):
            num_channels = [int(c) for c in num_channels]
        else:
            num_channels = [int(num_channels)]

        individual = bool(model_cfg.get("individual", False))
        feature_cols = list(data_cfg.get("feature_cols", []))
        target_feature_idx = (
            resolve_target_feature_index(cfg, feature_cols) if individual else None
        )

        return cls(
            input_size=num_features,
            num_channels=num_channels,
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            forecast_horizon=int(data_cfg.get("horizon", 1)),
            seq_len=int(data_cfg["window_size"]),
            individual=individual,
            target_feature_idx=target_feature_idx,
            decomp_kernel=int(model_cfg.get("decomp_kernel", 9)),
            use_weight_norm=bool(model_cfg.get("use_weight_norm", False)),
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "variant": "individual_tcn_decomp" if self.individual else "shared_tcn_decomp",
            "input_size": self.input_size,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "individual": self.individual,
            "target_feature_idx": self.target_feature_idx,
            "decomp_kernel": self.decomp_kernel,
            "use_weight_norm": self.use_weight_norm,
            "task_type": self.task_type,
            "forecast_horizon": self.forecast_horizon,
            "output_size": self.output_size,
            "num_params": sum(p.numel() for p in self.parameters()),
        }
