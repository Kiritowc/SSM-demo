import io
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from cv.core.backbone import *


_vault = MoEn()


@dataclass(frozen=True)
class BackboneArtifactCoordinate:
    structure_path: Optional[str] = None
    weight_path: Optional[str] = None


@dataclass(frozen=True)
class BackboneManifest:
    model_name: str
    dialect: str
    artifact_coordinate: BackboneArtifactCoordinate
    net_config: Optional[Dict[str, Any]]
    feature_pyramid_channels: Tuple[int, int, int]


class BackboneDialectResolver:
    def materialize(self, model_name: str) -> Tuple[nn.Module, BackboneManifest]:
        if model_name in ssg_list:
            return self._materialize_embedded_mcunet(model_name)
        return self._materialize_symbolic_backbone(model_name)

    def _materialize_embedded_mcunet(
        self, model_name: str
    ) -> Tuple[nn.Module, BackboneManifest]:
        artifact_coordinate = BackboneArtifactCoordinate(
            structure_path=f"cv/core/backbone/ssg/{model_name}/S.bin",
            weight_path=f"cv/core/backbone/ssg/{model_name}/W.bin",
        )
        net_config = self._decrypt_json(artifact_coordinate.structure_path)
        net_config["model_name"] = model_name
        _ = NET_INFO.get(model_name, {})
        module = mcuNASNets.build_from_config(net_config)
        state_dict = self._decrypt_torch_blob(artifact_coordinate.weight_path)[
            "state_dict"
        ]
        self._load_compatible_state(module, state_dict)
        manifest = BackboneManifest(
            model_name=model_name,
            dialect="embedded-mcunet",
            artifact_coordinate=artifact_coordinate,
            net_config=net_config,
            feature_pyramid_channels=(0, 0, 0),
        )
        return module, manifest

    def _materialize_symbolic_backbone(
        self, model_name: str
    ) -> Tuple[nn.Module, BackboneManifest]:
        module = build_registered_backbone(model_name)
        blueprint = resolve_backbone_blueprint(model_name)
        manifest = BackboneManifest(
            model_name=model_name,
            dialect=f"registry::{blueprint.family}",
            artifact_coordinate=BackboneArtifactCoordinate(
                structure_path=blueprint.artifact.structure_path,
                weight_path=blueprint.artifact.weight_path,
            ),
            net_config=None,
            feature_pyramid_channels=(0, 0, 0),
        )
        return module, manifest

    def _decrypt_json(self, path: str) -> Dict[str, Any]:
        decrypted_data = _vault.de_model_to_memory(path)
        return json.loads(decrypted_data.decode("utf-8"))

    def _decrypt_torch_blob(self, path: str) -> Dict[str, Any]:
        decrypted_data = _vault.de_model_to_memory(path)
        buffer = io.BytesIO(decrypted_data)
        return torch.load(buffer, map_location="cpu")

    def _load_compatible_state(self, module: nn.Module, state_dict: Dict[str, Any]) -> None:
        model_dict = module.state_dict()
        compatible_state = {}
        for key, value in state_dict.items():
            if key in model_dict and np.shape(model_dict[key]) == np.shape(value):
                compatible_state[key] = value
        unload_keys = list(set(model_dict.keys()).difference(set(compatible_state.keys())))
        unload_rate = len(unload_keys) / len(model_dict) * 100
        print("unload_rate: ", unload_rate, "%")
        model_dict.update(compatible_state)
        module.load_state_dict(model_dict)


class FeatureTopologyCondenser:
    def resolve(
        self,
        backbone: nn.Module,
        input_width: int,
        input_height: int,
        override_ins: Optional[str] = None,
        override_ous: Optional[str] = None,
    ) -> Tuple[int, int, Tuple[int, int, int]]:
        if override_ins and override_ous:
            ingress = int(override_ins)
            egress = int(override_ous)
            pyramid_channels = (0, 0, 0)
            return ingress, egress, pyramid_channels

        backbone.eval()
        with torch.no_grad():
            p1, p2, p3 = backbone(torch.rand((1, 3, input_width, input_height)))
        pyramid_channels = (p1.shape[1], p2.shape[1], p3.shape[1])
        ingress = sum(pyramid_channels)
        egress = 128
        return ingress, egress, pyramid_channels


class DetectorSemanticForge:
    def __init__(self):
        self.resolver = BackboneDialectResolver()
        self.condenser = FeatureTopologyCondenser()

    def forge(
        self,
        model_name: str,
        input_width: int,
        input_height: int,
        override_ins: Optional[str] = None,
        override_ous: Optional[str] = None,
    ) -> Tuple[nn.Module, BackboneManifest, int, int]:
        backbone, manifest = self.resolver.materialize(model_name)
        ingress, egress, pyramid_channels = self.condenser.resolve(
            backbone=backbone,
            input_width=input_width,
            input_height=input_height,
            override_ins=override_ins,
            override_ous=override_ous,
        )
        manifest = BackboneManifest(
            model_name=manifest.model_name,
            dialect=manifest.dialect,
            artifact_coordinate=manifest.artifact_coordinate,
            net_config=manifest.net_config,
            feature_pyramid_channels=pyramid_channels,
        )
        return backbone, manifest, ingress, egress
