import torch
import torch.nn as nn
from torch import Tensor

from cv.paths import backbone_bin

from .nexus import (
    BackboneBlueprint,
    EncryptedArtifactSpec,
    EncryptedWeightTransducer,
    backbone_registry,
)


SSO_BLUEPRINTS = {
    "sso_a": BackboneBlueprint(
        identifier="sso_a",
        family="sso",
        stage_repeats=(2, 4, 2),
        stage_channels=(24, 36, 72, 144),
        notes=("shufflenetv2-family", "partial-pretrained"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("sso", "sso_a.bin")),
    ),
    "sso_b": BackboneBlueprint(
        identifier="sso_b",
        family="sso",
        stage_repeats=(3, 6, 3),
        stage_channels=(24, 36, 72, 144),
        notes=("shufflenetv2-family", "partial-pretrained"),
    ),
    "sso_c": BackboneBlueprint(
        identifier="sso_c",
        family="sso",
        stage_repeats=(2, 4, 2),
        stage_channels=(24, 48, 96, 192),
        notes=("shufflenetv2-family", "partial-pretrained"),
    ),
    "sso_d": BackboneBlueprint(
        identifier="sso_d",
        family="sso",
        stage_repeats=(3, 6, 3),
        stage_channels=(24, 48, 96, 192),
        notes=("shufflenetv2-family", "partial-pretrained"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("sso", "sso_d.bin")),
    ),
    "sso_e": BackboneBlueprint(
        identifier="sso_e",
        family="sso",
        stage_repeats=(4, 8, 4),
        stage_channels=(24, 48, 96, 192),
        notes=("shufflenetv2-family", "partial-pretrained"),
    ),
    "sso_f": BackboneBlueprint(
        identifier="sso_f",
        family="sso",
        stage_repeats=(4, 8, 4),
        stage_channels=(24, 72, 144, 288),
        notes=("shufflenetv2-family", "partial-pretrained"),
    ),
    "sso_g": BackboneBlueprint(
        identifier="sso_g",
        family="sso",
        stage_repeats=(4, 8, 4),
        stage_channels=(24, 96, 192, 384),
        notes=("shufflenetv2-family", "partial-pretrained"),
    ),
    "sso_h": BackboneBlueprint(
        identifier="sso_h",
        family="sso",
        stage_repeats=(4, 8, 4),
        stage_channels=(24, 116, 232, 464),
        notes=("shufflenetv2-family", "partial-pretrained"),
    ),
    "sso_i": BackboneBlueprint(
        identifier="sso_i",
        family="sso",
        stage_repeats=(4, 8, 4),
        stage_channels=(24, 176, 352, 704),
        notes=("shufflenetv2-family", "partial-pretrained"),
    ),
    "sso_j": BackboneBlueprint(
        identifier="sso_j",
        family="sso",
        stage_repeats=(4, 8, 4),
        stage_channels=(24, 244, 488, 976),
        notes=("shufflenetv2-family", "partial-pretrained"),
    ),
}


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()
        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, blueprint: BackboneBlueprint):
        super().__init__()
        self.blueprint = blueprint
        self.transducer = EncryptedWeightTransducer()
        stages_repeats = blueprint.stage_repeats
        stages_out_channels = blueprint.stage_channels
        input_channels = 3
        output_channels = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ["stage2", "stage3", "stage4"]
        for name, repeats, output_channels in zip(
            stage_names, stages_repeats, stages_out_channels[1:]
        ):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        self.transducer.hydrate_module(self, blueprint.artifact.weight_path)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.maxpool(x)
        p1 = self.stage2(x)
        p2 = self.stage3(p1)
        p3 = self.stage4(p2)
        return p1, p2, p3


def _forge_sso(identifier: str) -> ShuffleNetV2:
    return ShuffleNetV2(SSO_BLUEPRINTS[identifier])


for _identifier, _blueprint in SSO_BLUEPRINTS.items():
    backbone_registry.register(_blueprint, lambda ident=_identifier: _forge_sso(ident))


def sso_a():
    return _forge_sso("sso_a")


def sso_b():
    return _forge_sso("sso_b")


def sso_c():
    return _forge_sso("sso_c")


def sso_d():
    return _forge_sso("sso_d")


def sso_e():
    return _forge_sso("sso_e")


def sso_f():
    return _forge_sso("sso_f")


def sso_g():
    return _forge_sso("sso_g")


def sso_h():
    return _forge_sso("sso_h")


def sso_i():
    return _forge_sso("sso_i")


def sso_j():
    return _forge_sso("sso_j")
