from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ssm.paths import get_paths, repo_root


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value.replace("${SSM_ROOT}", str(repo_root())))
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@lru_cache
def load_platform_config() -> dict:
    path = get_paths().configs_platform / "platform.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)


def platform_services() -> dict:
    return load_platform_config().get("services", {})


def platform_binaries() -> dict:
    return load_platform_config().get("binaries", {})


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)


def merge_dicts(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = value
    return out
