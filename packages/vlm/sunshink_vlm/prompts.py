from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ssm_common.config import load_yaml
from ssm_common.paths import get_paths


@lru_cache
def _load_prompt(name: str) -> dict:
    path = get_paths().configs / "vlm" / "prompts" / f"{name}.yaml"
    if not path.exists():
        path = get_paths().configs / "vlm" / "prompts" / "default.yaml"
    return load_yaml(path)


def build_vlm_system_prompt(*, robot_toy_mode: bool = False) -> str:
    key = "robot_toy" if robot_toy_mode else "default"
    data = _load_prompt(key)
    return (data.get("system") or "").strip()


def build_vlm_rules(*, robot_toy_mode: bool = False) -> str:
    key = "robot_toy" if robot_toy_mode else "default"
    data = _load_prompt(key)
    return (data.get("rules") or "").strip()
