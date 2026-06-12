"""Configuration loading, merging, validation, and CLI parsing."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_RELATIVE = Path("configs/default.yaml")
MODELS_CONFIG_DIR = Path("configs/models")
TASKS_CONFIG_DIR = Path("configs/tasks")
DATA_CONFIG_DIR = Path("configs/data")


def get_project_root(start: Path | None = None) -> Path:
    """Resolve project root from a script path or current working directory."""
    if start is not None:
        return start.resolve().parent if start.name.endswith(".py") else start.resolve()
    return Path.cwd()


def get_default_config_path(project_root: Path | None = None) -> Path:
    """Return path to the default YAML config."""
    root = project_root or Path.cwd()
    return root / DEFAULT_CONFIG_RELATIVE


def resolve_config_paths(
    configs: list[str | Path] | None,
    project_root: Path | None = None,
) -> list[Path]:
    """Resolve explicit config paths, always keeping default.yaml as the base layer."""
    root = project_root or Path.cwd()
    default_path = get_default_config_path(root)
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default config not found: {default_path}. "
            "Create configs/default.yaml or pass --config explicitly."
        )

    paths: list[Path] = [default_path]
    if configs:
        for path in configs:
            resolved = Path(path)
            if resolved.resolve() != default_path.resolve():
                paths.append(resolved)
    return paths


def list_model_configs(project_root: Path | None = None) -> list[str]:
    """List available model config names under configs/models/."""
    root = project_root or Path.cwd()
    models_dir = root / MODELS_CONFIG_DIR
    if not models_dir.exists():
        return []
    return sorted(path.stem for path in models_dir.glob("*.yaml"))


def list_task_configs(project_root: Path | None = None) -> list[str]:
    """List available task config names under configs/tasks/."""
    root = project_root or Path.cwd()
    tasks_dir = root / TASKS_CONFIG_DIR
    if not tasks_dir.exists():
        return []
    return sorted(path.stem for path in tasks_dir.glob("*.yaml"))


def list_data_configs(project_root: Path | None = None) -> list[str]:
    """List available data config profiles under configs/data/."""
    root = project_root or Path.cwd()
    data_dir = root / DATA_CONFIG_DIR
    if not data_dir.exists():
        return []
    return sorted(path.stem for path in data_dir.glob("*.yaml"))


def _effective_selectors(
    cfg: dict[str, Any],
    overrides: list[str] | None,
) -> tuple[str, str, str | None]:
    """Resolve model/task/data selectors after CLI overrides are applied."""
    model_name = str(cfg.get("model", {}).get("name", "")).lower()
    task_type = str(cfg.get("task", {}).get("type", "")).lower()
    data_profile = cfg.get("data", {}).get("profile")

    if overrides:
        for item in overrides:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            value = _parse_override_value(value.strip())
            if key == "model.name":
                model_name = str(value).lower()
            elif key == "task.type":
                task_type = str(value).lower()
            elif key == "data.profile":
                data_profile = str(value).lower() if value is not None else None

    return model_name, task_type, data_profile


def _infer_data_profile(cfg: dict[str, Any], project_root: Path) -> str | None:
    """Infer data profile name from explicit profile or dataset path stem."""
    profile = cfg.get("data", {}).get("profile")
    if profile:
        return str(profile).lower()

    data_path = cfg.get("data", {}).get("path")
    if not data_path:
        return None

    stem = Path(data_path).stem.lower()
    candidate = project_root / DATA_CONFIG_DIR / f"{stem}.yaml"
    if candidate.exists():
        return stem

    for name in list_data_configs(project_root):
        if name in stem or stem in name:
            profile_path = project_root / DATA_CONFIG_DIR / f"{name}.yaml"
            if profile_path.exists():
                return name
    return None


def _resolve_fragment_paths(
    model_name: str,
    task_type: str,
    project_root: Path,
) -> list[Path]:
    """Resolve auto-loaded model/task config fragments."""
    paths: list[Path] = []

    if model_name:
        model_path = project_root / MODELS_CONFIG_DIR / f"{model_name}.yaml"
        if model_path.exists():
            paths.append(model_path)

    if task_type:
        task_path = project_root / TASKS_CONFIG_DIR / f"{task_type}.yaml"
        if task_path.exists():
            paths.append(task_path)

    return paths


def _resolve_data_fragment_path(
    data_profile: str | None,
    project_root: Path,
) -> Path | None:
    """Resolve auto-loaded data config fragment."""
    if not data_profile:
        return None
    data_path = project_root / DATA_CONFIG_DIR / f"{data_profile}.yaml"
    if data_path.exists():
        return data_path
    return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_by_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    """Set nested config value using dot-separated path."""
    keys = path.split(".")
    current = cfg
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = _parse_override_value(value)


def _parse_override_value(value: str) -> Any:
    """Parse CLI override string to Python value."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered == "null" or lowered == "none":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_override_value(item.strip()) for item in inner.split(",")]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


def load_config(
    config_paths: list[str | Path],
    overrides: list[str] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """
    Load and merge config layers in order:

    1. explicit paths (default.yaml is always included first)
    2. auto model fragment from ``model.name`` -> ``configs/models/<name>.yaml``
    3. auto data fragment from ``data.profile`` or inferred path -> ``configs/data/<profile>.yaml``
    4. auto task fragment from ``task.type`` -> ``configs/tasks/<type>.yaml`` (overrides data)
    5. CLI overrides
    """
    if not config_paths:
        raise ValueError("At least one config file is required.")

    root = project_root or Path.cwd()
    merged: dict[str, Any] = {}
    for path in config_paths:
        merged = _deep_merge(merged, load_yaml(path))

    model_name, task_type, data_profile = _effective_selectors(merged, overrides)

    if model_name:
        model_path = root / MODELS_CONFIG_DIR / f"{model_name}.yaml"
        if model_path.exists():
            merged = _deep_merge(merged, load_yaml(model_path))

    if data_profile is None:
        data_profile = _infer_data_profile(merged, root)
    data_fragment = _resolve_data_fragment_path(data_profile, root)
    if data_fragment is not None:
        merged = _deep_merge(merged, load_yaml(data_fragment))

    if task_type:
        task_path = root / TASKS_CONFIG_DIR / f"{task_type}.yaml"
        if task_path.exists():
            merged = _deep_merge(merged, load_yaml(task_path))

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override format (expected key=value): {item}")
            key, value = item.split("=", 1)
            _set_by_path(merged, key.strip(), value.strip())

    validate_config(merged, project_root=root)
    return merged


def validate_config(cfg: dict[str, Any], project_root: Path | None = None) -> None:
    """Validate required config fields."""
    root = project_root or Path.cwd()
    required_sections = ["task", "model", "data", "train"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")

    task_type = cfg["task"].get("type")
    if task_type not in ("forecast", "classify"):
        raise ValueError(f"Invalid task.type: {task_type}. Must be 'forecast' or 'classify'.")

    model_name = str(cfg["model"].get("name", "")).lower()
    if not model_name:
        raise ValueError("model.name is required.")

    available_models = list_model_configs(root)
    model_path = root / MODELS_CONFIG_DIR / f"{model_name}.yaml"
    if available_models and not model_path.exists():
        available = ", ".join(available_models)
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Create configs/models/{model_name}.yaml or choose from: {available}"
        )

    model_cfg = cfg["model"]
    if model_name in ("lstm", "gru"):
        for key in ("hidden_size", "num_layers"):
            if key not in model_cfg:
                raise ValueError(
                    f"Missing model.{key}. "
                    f"Add configs/models/{model_name}.yaml or set it in the merged config."
                )

    data_cfg = cfg["data"]
    for key in ("path", "feature_cols", "window_size", "horizon"):
        if key not in data_cfg:
            raise ValueError(f"Missing required data.{key}")

    if task_type == "classify":
        classify_cfg = data_cfg.get("classify", {})
        if not classify_cfg.get("enabled", False):
            raise ValueError("task.type is 'classify' but data.classify.enabled is false.")
        if "num_classes" not in classify_cfg:
            raise ValueError("data.classify.num_classes is required for classification.")

    train_cfg = cfg["train"]
    for key in ("epochs", "batch_size", "lr", "log_dir"):
        if key not in train_cfg:
            raise ValueError(f"Missing required train.{key}")


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Save resolved config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

