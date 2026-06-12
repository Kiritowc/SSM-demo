import torch
import torch.nn as nn
from pathlib import Path

from .layers import *
from cv.utils.tool import *
from .manifold import DetectorSemanticForge

_REPO_ROOT = Path(__file__).resolve().parents[2]
datasetCfg = LoadYaml(str(_REPO_ROOT / "artifacts/cv/runtime/self.yaml"))
input_shape = [datasetCfg.input_height, datasetCfg.input_width]



class Detector(nn.Module):
    def __init__(self, category_num, opt, load_param):
        super(Detector, self).__init__()
        forge = DetectorSemanticForge()
        self.backbone, self.backbone_manifest, ins, ous = forge.forge(
            model_name=opt.model,
            input_width=datasetCfg.input_width,
            input_height=datasetCfg.input_height,
            override_ins=opt.ins,
            override_ous=opt.ous,
        )
        print("backbone dialect: ", self.backbone_manifest.dialect)
        print("feature pyramid channels: ", self.backbone_manifest.feature_pyramid_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        print("model ins: ", ins, ", ous: ", ous)
        if opt.spp.lower().strip() == "spp":
            self.SPP = SPP(ins, ous)
        elif opt.spp.lower().strip() == "spp357":
            self.SPP = SPP357(ins, ous)
        elif opt.spp.lower().strip() == "spp5913":
            self.SPP = SPP5913(ins, ous)
        self.detect_head = DetectHead(ous, category_num)

    def forward(self, x):
        P1, P2, P3 = self.backbone(x)
        P3 = self.upsample(P3)
        P1 = self.avg_pool(P1)
        P = torch.cat((P1, P2, P3), dim=1)
        y = self.SPP(P)
        return self.detect_head(y)
