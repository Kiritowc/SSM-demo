"""CV module path helpers — repo-relative paths derived from cv/configs/default.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ssm.config import load_yaml
from ssm.paths import get_paths


def backbone_bin(*parts: str) -> str:
    return str(Path("cv/artifacts/backbones").joinpath(*parts))


def artifact_rel(*parts: str) -> str:
    return str(Path("cv/artifacts").joinpath(*parts))


def data_rel(*parts: str) -> str:
    return str(Path("cv/data").joinpath(*parts))


def resolve(rel: str) -> Path:
    return get_paths().root / rel


@lru_cache
def load_cv_default() -> dict:
    return load_yaml(get_paths().configs_cv / "default.yaml")


def deploy_config() -> dict[str, str]:
    deploy = load_cv_default()["deploy"]
    return {
        "engine": deploy["engine"],
        "onnx": deploy.get("onnx", artifact_rel("engines/robot_toy.onnx")),
        "marker": deploy["marker"],
    }


RUNTIME_YAML = artifact_rel("runtime/self.yaml")
RUNTIME_NAMES = artifact_rel("runtime/self.names")
RUNS_DIR = artifact_rel("runs")
ROBOT_TOY_RUN_BIN = artifact_rel("runs/robot_toy/run.bin")
ROBOT_TOY_BEST_BIN = artifact_rel("runs/robot_toy/best.bin")
ROBOT_TOY_ENGINE = artifact_rel("engines/robot_toy.engine")
ROBOT_TOY_CLASSES = data_rel("robot_toy/classes.txt")
SAMPLE_IMAGE = data_rel("robot_toy/images/train/7.jpg")
