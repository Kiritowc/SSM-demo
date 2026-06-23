import torch
import torch.nn as nn

from cv.paths import backbone_bin

from .nexus import (
    BackboneBlueprint,
    EncryptedArtifactSpec,
    EncryptedWeightTransducer,
    backbone_registry,
)


SSI_BLUEPRINTS = {
    "ssi_a": BackboneBlueprint(
        identifier="ssi_a",
        family="ssi",
        stage_repeats=(2, 4, 2),
        stage_channels=(-1, 24, 36, 72, 144),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
    "ssi_b": BackboneBlueprint(
        identifier="ssi_b",
        family="ssi",
        stage_repeats=(3, 6, 3),
        stage_channels=(-1, 24, 36, 72, 144),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
    "ssi_c": BackboneBlueprint(
        identifier="ssi_c",
        family="ssi",
        stage_repeats=(2, 4, 2),
        stage_channels=(-1, 24, 48, 96, 192),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
    "ssi_d": BackboneBlueprint(
        identifier="ssi_d",
        family="ssi",
        stage_repeats=(3, 6, 3),
        stage_channels=(-1, 24, 48, 96, 192),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
    "ssi_e": BackboneBlueprint(
        identifier="ssi_e",
        family="ssi",
        stage_repeats=(4, 8, 4),
        stage_channels=(-1, 24, 48, 96, 192),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
    "ssi_f": BackboneBlueprint(
        identifier="ssi_f",
        family="ssi",
        stage_repeats=(4, 8, 4),
        stage_channels=(-1, 24, 72, 144, 288),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
    "ssi_g": BackboneBlueprint(
        identifier="ssi_g",
        family="ssi",
        stage_repeats=(4, 8, 4),
        stage_channels=(-1, 24, 96, 192, 384),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
    "ssi_h": BackboneBlueprint(
        identifier="ssi_h",
        family="ssi",
        stage_repeats=(4, 8, 4),
        stage_channels=(-1, 24, 116, 232, 464),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
    "ssi_i": BackboneBlueprint(
        identifier="ssi_i",
        family="ssi",
        stage_repeats=(4, 8, 4),
        stage_channels=(-1, 24, 176, 352, 704),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
    "ssi_j": BackboneBlueprint(
        identifier="ssi_j",
        family="ssi",
        stage_repeats=(4, 8, 4),
        stage_channels=(-1, 24, 244, 488, 976),
        notes=("shufflenetv2-family", "shared-encrypted-root"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssi", "ssi.bin")),
    ),
}


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        pad = ksize // 2
        outputs = oup - inp
        self.branch_main = nn.Sequential(
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                ksize,
                stride,
                pad,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )
        self.branch_proj = (
            nn.Sequential(
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            )
            if stride == 2
            else None
        )

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        return torch.cat((self.branch_proj(old_x), self.branch_main(old_x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self, blueprint: BackboneBlueprint):
        super().__init__()
        self.blueprint = blueprint
        self.transducer = EncryptedWeightTransducer()
        stage_repeats = blueprint.stage_repeats
        stage_out_channels = blueprint.stage_channels
        input_channel = stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage, numrepeat in enumerate(stage_repeats):
            output_channel = stage_out_channels[idxstage + 2]
            stage_seq = []
            for i in range(numrepeat):
                if i == 0:
                    stage_seq.append(
                        ShuffleV2Block(
                            input_channel,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=2,
                        )
                    )
                else:
                    stage_seq.append(
                        ShuffleV2Block(
                            input_channel // 2,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=1,
                        )
                    )
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stage_seq))
        self.transducer.hydrate_module(self, blueprint.artifact.weight_path)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        p1 = self.stage2(x)
        p2 = self.stage3(p1)
        p3 = self.stage4(p2)
        return p1, p2, p3


def _forge_ssi(identifier: str) -> ShuffleNetV2:
    return ShuffleNetV2(SSI_BLUEPRINTS[identifier])


for _identifier, _blueprint in SSI_BLUEPRINTS.items():
    backbone_registry.register(_blueprint, lambda ident=_identifier: _forge_ssi(ident))


def ssi_a():
    return _forge_ssi("ssi_a")


def ssi_b():
    return _forge_ssi("ssi_b")


def ssi_c():
    return _forge_ssi("ssi_c")


def ssi_d():
    return _forge_ssi("ssi_d")


def ssi_e():
    return _forge_ssi("ssi_e")


def ssi_f():
    return _forge_ssi("ssi_f")


def ssi_g():
    return _forge_ssi("ssi_g")


def ssi_h():
    return _forge_ssi("ssi_h")


def ssi_i():
    return _forge_ssi("ssi_i")


def ssi_j():
    return _forge_ssi("ssi_j")
