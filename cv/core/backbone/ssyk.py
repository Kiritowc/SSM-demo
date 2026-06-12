import torch
import torch.nn as nn

from .nexus import (
    BackboneBlueprint,
    EncryptedArtifactSpec,
    EncryptedWeightTransducer,
    backbone_registry,
    collect_sequential_features,
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

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(
                Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


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


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            b, self.num_heads, self.key_dim * 2 + self.head_dim, n
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(
            v.reshape(b, c, h, w)
        )
        return self.proj(x)


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


SSYK_VARIANT_PROGRAMS = {
    "ssyk_a": {
        "feature_taps": (4, 6, 10),
        "layers": (
            ("conv", (3, 16, 3, 2)),
            ("conv", (16, 32, 3, 2)),
            ("c3k2", (32, 64, 1, False, 0.25)),
            ("conv", (64, 64, 3, 2)),
            ("c3k2", (64, 128, 1, False, 0.25)),
            ("conv", (128, 128, 3, 2)),
            ("c3k2", (128, 128, 1, True)),
            ("conv", (128, 256, 3, 2)),
            ("c3k2", (256, 256, 1, True)),
            ("sppf", (256, 256, 5)),
            ("c2psa", (256, 256, 1)),
        ),
        "artifact": "cv/core/backbone/ssyk/ssyk_a.bin",
    },
    "ssyk_b": {
        "feature_taps": (4, 6, 10),
        "layers": (
            ("conv", (3, 32, 3, 2)),
            ("conv", (32, 64, 3, 2)),
            ("c3k2", (64, 128, 1, False, 0.25)),
            ("conv", (128, 128, 3, 2)),
            ("c3k2", (128, 256, 1, False, 0.25)),
            ("conv", (256, 256, 3, 2)),
            ("c3k2", (256, 256, 1, True)),
            ("conv", (256, 512, 3, 2)),
            ("c3k2", (512, 512, 1, True)),
            ("sppf", (512, 512, 5)),
            ("c2psa", (512, 512, 1)),
        ),
        "artifact": "cv/core/backbone/ssyk/ssyk_b.bin",
    },
}


SSYK_BLUEPRINTS = {
    identifier: BackboneBlueprint(
        identifier=identifier,
        family="ssyk",
        feature_taps=program["feature_taps"],
        notes=("attention-family", "programmed-sequential"),
        artifact=EncryptedArtifactSpec(weight_path=program["artifact"]),
    )
    for identifier, program in SSYK_VARIANT_PROGRAMS.items()
}


def _materialize_ssyk_atom(opcode: str, args):
    if opcode == "conv":
        return Conv(*args)
    if opcode == "c3k2":
        return C3k2(*args)
    if opcode == "sppf":
        return SPPF(*args)
    if opcode == "c2psa":
        return C2PSA(*args)
    raise ValueError(f"Unknown ssyk opcode: {opcode}")


class SsykProgrammaticChassis(nn.Module):
    def __init__(self, blueprint: BackboneBlueprint):
        super().__init__()
        self.blueprint = blueprint
        self.transducer = EncryptedWeightTransducer()
        program = SSYK_VARIANT_PROGRAMS[blueprint.identifier]
        self.backbone = nn.Sequential(
            *(_materialize_ssyk_atom(opcode, args) for opcode, args in program["layers"])
        )
        self.transducer.hydrate_module(self, blueprint.artifact.weight_path)

    def forward(self, x):
        return collect_sequential_features(x, self.backbone, self.blueprint.feature_taps)


def _forge_ssyk(identifier: str) -> SsykProgrammaticChassis:
    return SsykProgrammaticChassis(SSYK_BLUEPRINTS[identifier])


for _identifier, _blueprint in SSYK_BLUEPRINTS.items():
    backbone_registry.register(_blueprint, lambda ident=_identifier: _forge_ssyk(ident))


def ssyk_a():
    return _forge_ssyk("ssyk_a")


def ssyk_b():
    return _forge_ssyk("ssyk_b")

