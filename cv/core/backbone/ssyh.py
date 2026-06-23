import torch
import torch.nn as nn

from cv.paths import backbone_bin

from .nexus import (
    BackboneBlueprint,
    EncryptedArtifactSpec,
    EncryptedWeightTransducer,
    backbone_registry,
)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SsyhChassis(nn.Module):
    def __init__(self, blueprint: BackboneBlueprint):
        super().__init__()
        self.blueprint = blueprint
        self.transducer = EncryptedWeightTransducer()
        c1, c2, c3, c4, c5, c6 = blueprint.stage_channels
        r1, r2, r3, r4 = blueprint.stage_repeats

        self.stem = nn.Sequential(Conv(3, c1, 3, 2), Conv(c1, c2, 3, 2))
        self.stage1 = self._make_transition_stage(c2, c3, r1)
        self.stage2 = self._make_transition_stage(c3, c4, r2)
        self.stage3 = self._make_transition_stage(c4, c5, r3)
        self.stage4 = nn.Sequential(C2f(c5, c6, n=r4, shortcut=True))

        self.transducer.hydrate_module(self, blueprint.artifact.weight_path)

    def _make_transition_stage(self, ingress: int, egress: int, repeats: int) -> nn.Sequential:
        return nn.Sequential(
            C2f(ingress, ingress, n=repeats, shortcut=True),
            Conv(ingress, egress, 3, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p3 = self.stage4(p3)
        return p1, p2, p3


SSYH_BLUEPRINTS = {
    "ssyh_a": BackboneBlueprint(
        identifier="ssyh_a",
        family="ssyh",
        stage_repeats=(1, 2, 2, 1),
        stage_channels=(16, 32, 64, 128, 256, 256),
        notes=("c2f-yoloish", "encrypted-family"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssyh", "ssyh_a.bin")),
    ),
    "ssyh_b": BackboneBlueprint(
        identifier="ssyh_b",
        family="ssyh",
        stage_repeats=(1, 2, 2, 1),
        stage_channels=(32, 64, 128, 256, 512, 512),
        notes=("c2f-yoloish", "encrypted-family"),
        artifact=EncryptedArtifactSpec(weight_path=backbone_bin("ssyh", "ssyh_b.bin")),
    ),
}


def _forge_ssyh(identifier: str) -> SsyhChassis:
    return SsyhChassis(SSYH_BLUEPRINTS[identifier])


for _identifier, _blueprint in SSYH_BLUEPRINTS.items():
    backbone_registry.register(_blueprint, lambda ident=_identifier: _forge_ssyh(ident))


def ssyh_a():
    return _forge_ssyh("ssyh_a")


def ssyh_b():
    return _forge_ssyh("ssyh_b")


def ssyh_c():
    return _forge_ssyh("ssyh_c")


def ssyh_d():
    return _forge_ssyh("ssyh_d")


def ssyh_e():
    return _forge_ssyh("ssyh_e")
