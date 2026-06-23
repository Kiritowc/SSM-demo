"""EtsH: OmniFusion heavy conv forecaster (features_only)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ets.models.base import BaseForecastModel
from ets.models.layers.tcn import Chomp1d, TemporalBlock


def _grouped_conv_groups(in_channels: int, max_groups: int = 8) -> int:
    """Pick the largest group count dividing in_channels (generic, not hardcoded)."""
    for groups in range(min(in_channels, max_groups), 0, -1):
        if in_channels % groups == 0:
            return groups
    return 1


def _horizon_n_groups(horizon: int, preferred: int = 8) -> int:
    """Pick a group count that divides horizon (generic long-horizon head)."""
    for groups in range(min(horizon, preferred), 0, -1):
        if horizon % groups == 0:
            return groups
    return 1


def _format_output(
    logits: torch.Tensor,
    forecast_horizon: int,
    output_size: int,
) -> torch.Tensor:
    if forecast_horizon == 1:
        return logits.view(-1, output_size)
    return logits.view(-1, forecast_horizon, output_size)


class RevIN(nn.Module):
    """Reversible instance norm on encoder input (PatchTST / ModernTCN)."""

    def __init__(self, num_features: int, affine: bool = True) -> None:
        super().__init__()
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - mean) / std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        scale = x.mean(dim=-1)
        scale = self.fc(scale).unsqueeze(-1)
        return x * scale


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_channel_ln: bool = False,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.chomp = Chomp1d(padding)
        self.norm: nn.Module = (
            ChannelLayerNorm1d(channels)
            if use_channel_ln
            else nn.BatchNorm1d(channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.chomp(out)
        out = self.pointwise(out)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + x


class GatedTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.gate_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp = Chomp1d(padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filt = torch.tanh(self.chomp(self.filter_conv(x)))
        gate = torch.sigmoid(self.chomp(self.gate_conv(x)))
        out = self.dropout(filt * gate)
        return out + x


class DilatedTCNStack(nn.Module):
    """TCN stack with explicit dilation schedule."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(dilations) != len(num_channels):
            raise ValueError("dilations must match num_channels length")
        layers = []
        in_ch = num_inputs
        for out_ch, dilation in zip(num_channels, dilations):
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                )
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LayerNormTCNStack(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        in_ch = num_inputs
        for i, out_ch in enumerate(num_channels):
            dilation = 2**i
            blocks.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                    use_weight_norm=False,
                )
            )
            blocks.append(nn.LayerNorm(out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if isinstance(block, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = block(x)
                x = x.transpose(1, 2)
            else:
                x = block(x)
        return x


class PreMixTCNStack(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = num_inputs
        for i, out_ch in enumerate(num_channels):
            layers.append(nn.Conv1d(in_ch, in_ch, 1))
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                    use_weight_norm=False,
                )
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiScaleTCN(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                DilatedTCNStack(
                    num_inputs,
                    num_channels,
                    kernel_size,
                    dilations=[d] * len(num_channels),
                    dropout=dropout,
                )
                for d in dilations
            ]
        )
        self.out_channels = num_channels[-1] * len(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        return torch.cat(outs, dim=1)


class MultiKernelConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels: list[int]) -> None:
        super().__init__()
        branch_ch = max(out_ch // len(kernels), 1)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_ch, branch_ch, k, padding=k // 2)
                for k in kernels
            ]
        )
        self.proj = nn.Conv1d(branch_ch * len(kernels), out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [F.relu(branch(x)) for branch in self.branches]
        return self.proj(torch.cat(parts, dim=1))


class CausalMultiKernelConv(nn.Module):
    """Multi-kernel parallel conv with causal padding (no future leak)."""

    def __init__(self, in_ch: int, out_ch: int, kernels: list[int]) -> None:
        super().__init__()
        branch_ch = max(out_ch // len(kernels), 1)
        self.branches = nn.ModuleList()
        for k in kernels:
            padding = k - 1
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, branch_ch, k, padding=padding),
                    Chomp1d(padding),
                )
            )
        self.proj = nn.Conv1d(branch_ch * len(kernels), out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [F.gelu(branch(x)) for branch in self.branches]
        return self.proj(torch.cat(parts, dim=1))


class ConvFFN1d(nn.Module):
    """Inverted bottleneck FFN with causal depthwise (ModernTCN-style)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        expansion: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        padding = kernel_size - 1
        self.expand = nn.Conv1d(channels, hidden, 1)
        self.dw = nn.Conv1d(
            hidden, hidden, kernel_size, padding=padding, groups=hidden
        )
        self.chomp = Chomp1d(padding)
        self.project = nn.Conv1d(hidden, channels, 1)
        self.norm = ChannelLayerNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        y = F.gelu(self.chomp(self.dw(y)))
        y = self.project(y)
        y = self.norm(y)
        return x + self.dropout(y)


class ChannelLayerNorm1d(nn.Module):
    """LayerNorm over channel dim for (B, C, T) tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.norm(y)
        return y.transpose(1, 2)


class CausalTemporalAttnPool(nn.Module):
    """Causal depthwise conv pooling (pure conv, no attention matmul)."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size - 1
        self.dw = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, groups=channels
        )
        self.chomp = Chomp1d(padding)
        self.pw = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.pw(F.relu(self.chomp(self.dw(x))))
        weights = F.softmax(score, dim=-1)
        return (x * weights).sum(dim=-1)


class DynamicBranchGate(nn.Module):
    """Input-dependent branch weights (SKNet 思想，纯 1×1 conv)."""

    def __init__(self, channels: int, n_branches: int = 3) -> None:
        super().__init__()
        self.gate = nn.Conv1d(channels, n_branches, kernel_size=1)

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        ctx = sum(branch.mean(dim=-1, keepdim=True) for branch in branches) / len(branches)
        weights = F.softmax(self.gate(ctx), dim=1)
        out = branches[0] * weights[:, 0:1]
        for i in range(1, len(branches)):
            out = out + branches[i] * weights[:, i : i + 1]
        return out


class ParallelFullMixBlock(nn.Module):
    """
    全宽并行膨胀 depthwise 分支求和 + SE（优于通道切分 concat）。
    借鉴 ModernTCN / Inception-TCN 多感受野融合。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in dilations:
            padding = (kernel_size - 1) * dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=channels,
                    ),
                    Chomp1d(padding),
                    nn.Conv1d(channels, channels, 1),
                    nn.GELU(),
                )
            )
        self.se = SqueezeExcite1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = sum(branch(x) for branch in self.branches)
        y = self.se(y)
        return x + self.dropout(y)


class LargeKernelDWBlock(nn.Module):
    """大核因果 depthwise（ModernTCN 风格，左侧 padding + Chomp 无未来泄露）。"""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size - 1
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
        )
        self.chomp = Chomp1d(padding)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm = ChannelLayerNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.chomp(self.dw(x))
        y = self.pw(y)
        y = self.norm(y)
        y = F.gelu(y)
        return x + self.dropout(y)


class InvertedBottleneckGLU(nn.Module):
    """倒瓶颈 + WaveNet 门控（参数效率高于双路全宽 GLU）。"""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        padding = (kernel_size - 1) * dilation
        self.expand = nn.Conv1d(channels, hidden * 2, 1)
        self.dw = nn.Conv1d(
            hidden * 2,
            hidden * 2,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden * 2,
        )
        self.chomp = Chomp1d(padding)
        self.project = nn.Conv1d(hidden, channels, 1)
        self.norm = ChannelLayerNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        y = self.chomp(self.dw(y))
        a, b = y.chunk(2, dim=1)
        y = torch.tanh(a) * torch.sigmoid(b)
        y = self.project(y)
        y = self.norm(y)
        return x + self.dropout(y)


class MixtureDilatedBlock(nn.Module):
    """Depthwise multi-dilation mix + SE + residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        n = len(dilations)
        per = channels // n
        rem = channels % n
        self.branches = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            out_ch = per + (1 if i < rem else 0)
            padding = (kernel_size - 1) * dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=channels,
                    ),
                    Chomp1d(padding),
                    nn.Conv1d(channels, out_ch, 1),
                    nn.ReLU(),
                )
            )
        self.se = SqueezeExcite1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([branch(x) for branch in self.branches], dim=1)
        y = self.se(y)
        return x + self.dropout(y)


class ScaleBranch(nn.Module):
    """Lightweight scale-specific depthwise stack."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        dropout: float,
        use_channel_ln: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for dilation in dilations:
            layers.append(
                DepthwiseSeparableBlock(
                    channels, kernel_size, dilation, dropout, use_channel_ln=use_channel_ln
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyramidSkip(nn.Module):
    """Coarse half-resolution path upsampled back and added."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        mid = max(out_ch // 2, 24)
        padding = kernel_size - 1
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, mid, kernel_size, stride=2, padding=padding),
            Chomp1d(padding),
        )
        self.body = nn.Sequential(
            DepthwiseSeparableBlock(
                mid, kernel_size, dilation=1, dropout=dropout, use_channel_ln=True
            ),
            DepthwiseSeparableBlock(
                mid, kernel_size, dilation=2, dropout=dropout, use_channel_ln=True
            ),
            nn.Conv1d(mid, out_ch, 1),
        )

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.down(x)
        y = self.body(y)
        return F.interpolate(y, size=target_len, mode="linear", align_corners=False)


class PatchConvContext(nn.Module):
    """非重叠 patch 卷积嵌入（优于均值 patch，借鉴 PatchTST 卷积化）。"""

    def __init__(self, channels: int, patch_len: int = 8) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.embed = nn.Conv1d(channels, channels, patch_len, stride=patch_len)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        p = self.patch_len
        n = t // p
        if n == 0:
            return torch.zeros_like(x)
        coarse = self.proj(self.embed(x[:, :, : n * p]))
        return F.interpolate(coarse, size=t, mode="linear", align_corners=False)


class PatchContextBias(nn.Module):
    """Patch means → coarse temporal bias broadcast (no flatten head)."""

    def __init__(self, channels: int, patch_len: int = 8) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        p = self.patch_len
        n = t // p
        if n == 0:
            return torch.zeros_like(x)
        patches = x[:, :, : n * p].reshape(b, c, n, p).mean(dim=-1)
        coarse = self.proj(patches)
        return F.interpolate(coarse, size=t, mode="linear", align_corners=False)


class LocalWindowPool(nn.Module):
    """Causal depthwise conv over trailing window → pooled local context."""

    def __init__(self, channels: int, window: int = 12, kernel_size: int = 3) -> None:
        super().__init__()
        self.window = window
        padding = kernel_size - 1
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels),
            Chomp1d(padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tail = x[:, :, -self.window :]
        y = self.conv(tail)
        return y.mean(dim=-1)


class HorizonGroupHead(nn.Module):
    """Shared trunk + grouped step heads for long horizon."""

    def __init__(
        self,
        in_dim: int,
        horizon: int,
        n_groups: int = 8,
        trunk_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if horizon % n_groups != 0:
            raise ValueError(f"horizon {horizon} must be divisible by n_groups {n_groups}")
        self.steps_per_group = horizon // n_groups
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.group_heads = nn.ModuleList(
            [nn.Linear(trunk_dim, self.steps_per_group) for _ in range(n_groups)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.group_heads], dim=-1)


class OmniFusionEncoder(nn.Module):
    """
    a21 OmniFusion 定版 — pure-causal-conv generic industrial forecaster.

    Design references (conv-only lineage):
    - TCN / WaveNet: causal dilated conv + gated activations
    - ModernTCN: large-kernel DW, inverted bottleneck, ConvFFN, channel LayerNorm
    - PatchTST / RevIN: reversible instance normalization + patch conv context
    - Inception / SKNet: multi-scale parallel branches + dynamic gating
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        stem_dim: int = 56,
        width: int = 72,
        readout_dim: int = 128,
    ) -> None:
        super().__init__()
        self.width = width
        self.out_dim = readout_dim
        k = kernel_size
        groups = _grouped_conv_groups(input_size)
        stem_out = ((stem_dim + groups - 1) // groups) * groups
        large_k = min(31, seq_len if seq_len % 2 == 1 else seq_len - 1)
        if large_k < 7:
            large_k = 7
        stem_pad = k - 1

        self.revin = RevIN(input_size, affine=True)
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_size, stem_out, k, padding=stem_pad, groups=groups
            ),
            Chomp1d(stem_pad),
            nn.Conv1d(stem_out, stem_dim, 1),
            nn.GELU(),
            ChannelLayerNorm1d(stem_dim),
        )
        self.raw_skip = nn.Conv1d(input_size, width, 1)
        self.input_skip = nn.Conv1d(stem_dim, width, 1)

        ln = True
        self.short_branch = ScaleBranch(stem_dim, k, (1, 2), dropout, use_channel_ln=ln)
        self.med_branch = ScaleBranch(stem_dim, k, (4, 8), dropout, use_channel_ln=ln)
        self.wide_branch = nn.Sequential(
            CausalMultiKernelConv(stem_dim, stem_dim, kernels=[3, 5, 7]),
            DepthwiseSeparableBlock(stem_dim, k, dilation=16, dropout=dropout, use_channel_ln=ln),
        )
        self.branch_gate = DynamicBranchGate(stem_dim, n_branches=3)
        self.stem_fuse = nn.Sequential(
            nn.Conv1d(stem_dim, width, 1),
            nn.GELU(),
            SqueezeExcite1d(width),
        )

        self.mix_blocks = nn.ModuleList(
            [
                ParallelFullMixBlock(width, k, (1, 4, 16), dropout),
                ParallelFullMixBlock(width, k, (2, 8, 32), dropout),
                LargeKernelDWBlock(width, large_k, dropout),
                InvertedBottleneckGLU(width, k, dilation=2, expansion=2, dropout=dropout),
                ConvFFN1d(width, kernel_size=k, expansion=2, dropout=dropout),
                TemporalBlock(
                    width,
                    width,
                    k,
                    stride=1,
                    dilation=1,
                    padding=(k - 1),
                    dropout=dropout,
                    use_weight_norm=False,
                ),
            ]
        )
        self.pyramid = PyramidSkip(stem_dim, width, k, dropout)
        self.patch_ctx = PatchConvContext(width, patch_len=min(8, seq_len))

        self.local_window = LocalWindowPool(width, window=min(12, seq_len), kernel_size=3)
        self.attn_pool = CausalTemporalAttnPool(width, kernel_size=5)
        self.readout_proj = nn.Sequential(
            nn.Linear(width * 5, readout_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x_seq = x.transpose(1, 2)
        x_seq = self.revin(x_seq)
        x = x_seq.transpose(1, 2)

        raw_skip = self.raw_skip(x)
        stem = self.stem(x)
        skip = self.input_skip(stem)

        branches = [
            self.short_branch(stem),
            self.med_branch(stem),
            self.wide_branch(stem),
        ]
        mixed = self.branch_gate(branches)
        y = self.stem_fuse(mixed)

        coarse = self.pyramid(stem, y.shape[-1])
        y = y + coarse + self.patch_ctx(y)

        for block in self.mix_blocks:
            y = block(y)
        y = y + skip + raw_skip

        local_last = y[:, :, -1]
        local_win = self.local_window(y)
        attn = self.attn_pool(y)
        avg = y.mean(dim=-1)
        mx = y.max(dim=-1).values
        feat = torch.cat([local_last, local_win, attn, avg, mx], dim=-1)
        return self.readout_proj(feat)


class EtsHModel(BaseForecastModel):
    """OmniFusion encoder + grouped horizon head (~318K params)."""

    def __init__(
        self,
        input_size: int,
        forecast_horizon: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        seq_len: int = 96,
        output_size: int = 1,
        stem_dim: int | None = None,
        width: int | None = None,
        readout_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.num_channels = num_channels or [32, 32]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.seq_len = seq_len
        self.output_size = output_size
        self.c_out = output_size * forecast_horizon

        c_last = int(self.num_channels[-1])
        resolved_width = width if width is not None else max(c_last * 2, 72)
        resolved_stem = stem_dim if stem_dim is not None else max(c_last + 24, 56)

        self.encoder = OmniFusionEncoder(
            input_size=input_size,
            seq_len=seq_len,
            kernel_size=kernel_size,
            dropout=dropout,
            stem_dim=resolved_stem,
            width=resolved_width,
            readout_dim=readout_dim,
        )
        n_groups = _horizon_n_groups(forecast_horizon, preferred=8)
        self.head = HorizonGroupHead(
            in_dim=readout_dim,
            horizon=forecast_horizon,
            n_groups=n_groups,
            trunk_dim=128,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x.transpose(1, 2))
        logits = self.head(feat)
        return _format_output(logits, self.forecast_horizon, self.output_size)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_features: int) -> EtsHModel:
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        num_channels = model_cfg.get("num_channels", [32, 32])
        if isinstance(num_channels, (list, tuple)):
            num_channels = [int(c) for c in num_channels]
        else:
            num_channels = [int(num_channels)]
        return cls(
            input_size=num_features,
            forecast_horizon=int(data_cfg.get("horizon", 1)),
            num_channels=num_channels,
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            seq_len=int(data_cfg["window_size"]),
            output_size=int(model_cfg.get("output_size", 1)),
            stem_dim=model_cfg.get("stem_dim"),
            width=model_cfg.get("width"),
            readout_dim=int(model_cfg.get("readout_dim", 128)),
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "task_type": self.task_type,
            "forecast_horizon": self.forecast_horizon,
            "output_size": self.output_size,
            "description": "OmniFusion heavy conv forecaster (ets_h)",
        }
