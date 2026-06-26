import io
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .mi import MoEn


@dataclass(frozen=True)
class EncryptedArtifactSpec:
    weight_path: Optional[str] = None
    structure_path: Optional[str] = None


@dataclass(frozen=True)
class BackboneBlueprint:
    identifier: str
    family: str
    feature_taps: Tuple[int, ...] = ()
    stage_repeats: Tuple[int, ...] = ()
    stage_channels: Tuple[int, ...] = ()
    notes: Tuple[str, ...] = ()
    artifact: EncryptedArtifactSpec = field(default_factory=EncryptedArtifactSpec)


class EncryptedWeightTransducer:
    def __init__(self):
        self.decryptor = MoEn()
        self.decryptor.cipher_suite = MoEn.load_cipher_suite()

    def load_state_dict(self, weight_path: str):
        decrypted_data = self.decryptor.de_model_to_memory(weight_path)
        buffer = io.BytesIO(decrypted_data)
        return torch.load(buffer, map_location=torch.device("cpu"))

    def graft_state(self, module: nn.Module, state_dict, verbose: bool = True) -> nn.Module:
        model_dict = module.state_dict()
        weight_dict = {}
        for key, value in state_dict.items():
            if key in model_dict and np.shape(model_dict[key]) == np.shape(value):
                weight_dict[key] = value
        unload_keys = list(set(model_dict.keys()).difference(set(weight_dict.keys())))
        if verbose:
            unload_rate = len(unload_keys) / len(model_dict) * 100
            print("unload_rate: ", unload_rate, "%")
        model_dict.update(weight_dict)
        module.load_state_dict(model_dict)
        return module

    def hydrate_module(self, module: nn.Module, weight_path: Optional[str], verbose: bool = True) -> nn.Module:
        if not weight_path:
            return module
        state_dict = self.load_state_dict(weight_path)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return self.graft_state(module, state_dict, verbose=verbose)


class BackboneRegistry:
    def __init__(self):
        self._constructors: Dict[str, Callable[[], nn.Module]] = {}
        self._blueprints: Dict[str, BackboneBlueprint] = {}

    def register(self, blueprint: BackboneBlueprint, constructor: Callable[[], nn.Module]) -> None:
        self._blueprints[blueprint.identifier] = blueprint
        self._constructors[blueprint.identifier] = constructor

    def build(self, identifier: str) -> nn.Module:
        if identifier not in self._constructors:
            raise KeyError(f"Unknown backbone identifier: {identifier}")
        return self._constructors[identifier]()

    def blueprint(self, identifier: str) -> BackboneBlueprint:
        if identifier not in self._blueprints:
            raise KeyError(f"Unknown backbone blueprint: {identifier}")
        return self._blueprints[identifier]

    def identifiers(self) -> Tuple[str, ...]:
        return tuple(sorted(self._constructors.keys()))


backbone_registry = BackboneRegistry()


def collect_sequential_features(
    x: torch.Tensor,
    layers: Iterable[nn.Module],
    taps: Tuple[int, ...],
    emit_trace: bool = False,
):
    features = []
    for index, layer in enumerate(layers):
        x = layer(x)
        if emit_trace:
            print(f"feature@{index}: ", tuple(x.shape))
        if index in taps:
            features.append(x)
    return tuple(features)
