import torch
import torch.nn as nn

from cv.paths import backbone_bin

from .nexus import (
    BackboneBlueprint,
    EncryptedArtifactSpec,
    EncryptedWeightTransducer,
    backbone_registry,
    collect_sequential_features,
)


def activation_function(act="RE"):
    if act == "RE":
        return nn.ReLU6(inplace=True)
    if act == "GE":
        return nn.GELU()
    if act == "SI":
        return nn.SiLU()
    if act == "EL":
        return nn.ELU()
    return nn.Hardswish()


class mn_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act="RE", p=None, g=1, d=1):
        super().__init__()
        padding = p if p is not None else (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, dilation=d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation_function(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MobileNetV3_BLOCK(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        super().__init__()
        c_mid = e if e is not None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else []
        features.extend(
            [
                mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
                nn.Conv2d(c_mid, c2, 1),
                nn.BatchNorm2d(c2),
            ]
        )
        self.layers = nn.Sequential(*features)

    def forward(self, x):
        return x + self.layers(x) if self.residual else self.layers(x)


SSYL_VARIANT_PROGRAMS = {
    "ssyl_a": {
        "feature_taps": (2, 7, 11),
        "stem": ((3, 16, 3, 2, "SI"), (16, 16, 1, 1, "SI")),
        "blocks": (
            (16, 16, 3, 16, False, "SI", 2, False),
            (16, 32, 3, 96, False, "SI", 2, True),
            (32, 32, 3, 96, False, "SI", 1, True),
            (32, 64, 5, 96, True, "SI", 2, True),
            (64, 64, 5, 192, True, "SI", 1, True),
            (64, 64, 5, 192, True, "SI", 1, True),
            (64, 64, 5, 192, True, "SI", 1, True),
            (64, 64, 5, 192, True, "SI", 1, True),
            (64, 96, 5, 576, True, "SI", 2, True),
            (96, 96, 5, 576, True, "SI", 1, True),
            (96, 96, 5, 576, True, "SI", 1, True),
            (96, 128, 5, 576, True, "SI", 1, True),
        ),
        "artifact": backbone_bin("ssyl", "ssyl_a.bin"),
    },
    "ssyl_b": {
        "feature_taps": (2, 7, 11),
        "stem": ((3, 24, 3, 2, "SI"), (24, 24, 1, 1, "SI")),
        "blocks": (
            (24, 24, 3, 24, False, "SI", 2, False),
            (24, 48, 3, 128, False, "SI", 2, True),
            (48, 48, 3, 128, False, "SI", 1, True),
            (48, 88, 5, 128, True, "SI", 2, True),
            (88, 88, 5, 256, True, "SI", 1, True),
            (88, 88, 5, 256, True, "SI", 1, True),
            (88, 88, 5, 256, True, "SI", 1, True),
            (88, 96, 5, 256, True, "SI", 1, True),
            (96, 128, 5, 768, True, "SI", 2, True),
            (128, 128, 5, 768, True, "SI", 1, True),
            (128, 128, 5, 768, True, "SI", 1, True),
            (128, 192, 5, 768, True, "SI", 1, True),
        ),
        "artifact": backbone_bin("ssyl", "ssyl_b.bin"),
    },
    "ssyl_c": {
        "feature_taps": (3, 9, 14),
        "stem": ((3, 24, 3, 2, "SI"), (24, 24, 1, 1, "SI")),
        "blocks": (
            (24, 24, 3, 24, False, "SI", 2, False),
            (24, 48, 3, 128, False, "SI", 2, True),
            (48, 48, 3, 128, False, "SI", 1, True),
            (48, 48, 3, 128, False, "SI", 1, True),
            (48, 88, 5, 128, True, "SI", 2, True),
            (88, 88, 5, 256, True, "SI", 1, True),
            (88, 88, 5, 256, True, "SI", 1, True),
            (88, 88, 5, 256, True, "SI", 1, True),
            (88, 88, 5, 256, True, "SI", 1, True),
            (88, 96, 5, 256, True, "SI", 1, True),
            (96, 128, 5, 768, True, "SI", 2, True),
            (128, 128, 5, 768, True, "SI", 1, True),
            (128, 128, 5, 768, True, "SI", 1, True),
            (128, 128, 5, 768, True, "SI", 1, True),
            (128, 192, 5, 768, True, "SI", 1, True),
        ),
        "artifact": backbone_bin("ssyl", "ssyl_c.bin"),
    },
}


SSYL_BLUEPRINTS = {
    identifier: BackboneBlueprint(
        identifier=identifier,
        family="ssyl",
        feature_taps=program["feature_taps"],
        notes=("mobilenetv3-family", "blueprint-driven"),
        artifact=EncryptedArtifactSpec(weight_path=program["artifact"]),
    )
    for identifier, program in SSYL_VARIANT_PROGRAMS.items()
}


class SsyllableChassis(nn.Module):
    def __init__(self, blueprint: BackboneBlueprint):
        super().__init__()
        self.blueprint = blueprint
        self.transducer = EncryptedWeightTransducer()
        program = SSYL_VARIANT_PROGRAMS[blueprint.identifier]
        self.stem = nn.Sequential(
            *(mn_conv(c1, c2, k, s, act=act) for c1, c2, k, s, act in program["stem"])
        )
        self.blocks = nn.Sequential(
            *(
                MobileNetV3_BLOCK(
                    c1,
                    c2,
                    k=k,
                    e=e,
                    sa=sa,
                    act=act,
                    stride=stride,
                    pw=pw,
                )
                for c1, c2, k, e, sa, act, stride, pw in program["blocks"]
            )
        )
        self.transducer.hydrate_module(self, blueprint.artifact.weight_path)

    def forward(self, x):
        x = self.stem(x)
        return collect_sequential_features(x, self.blocks, self.blueprint.feature_taps)


def _forge_ssyl(identifier: str) -> SsyllableChassis:
    return SsyllableChassis(SSYL_BLUEPRINTS[identifier])


for _identifier, _blueprint in SSYL_BLUEPRINTS.items():
    backbone_registry.register(_blueprint, lambda ident=_identifier: _forge_ssyl(ident))


def ssyl_a():
    return _forge_ssyl("ssyl_a")


def ssyl_b():
    return _forge_ssyl("ssyl_b")


def ssyl_c():
    return _forge_ssyl("ssyl_c")
