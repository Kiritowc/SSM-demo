"""Ensure SSM_ROOT is set and repo root is importable (standalone scripts)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SSM_CONDA_ENV = "ssm"


def bootstrap_repo(start: Path | None = None) -> Path:
    """Set SSM_ROOT, prepend repo to sys.path; return absolute repo root."""
    if start is None:
        root = Path(__file__).resolve().parents[1]
    else:
        root = start.resolve()
    os.environ.setdefault("SSM_ROOT", str(root))
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _conda_roots() -> list[Path]:
    roots: list[Path] = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        roots.append(Path(conda_exe).resolve().parent.parent)
    roots.append(Path.home() / "miniconda3")
    roots.append(Path.home() / "anaconda3")
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def conda_python(env_name: str | None = None) -> Path | None:
    """Resolve the Python executable for the project conda env."""
    env = env_name or os.environ.get("SSM_CONDA_ENV", SSM_CONDA_ENV)
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and Path(prefix).name == env:
        active = Path(prefix) / "bin" / "python"
        if active.is_file():
            return active

    for root in _conda_roots():
        candidate = root / "envs" / env / "bin" / "python"
        if candidate.is_file():
            return candidate
    return None


def runtime_python() -> Path:
    """Preferred Python for SSDet/ETS/VLM scripts and cron jobs."""
    override = os.environ.get("SSM_PYTHON")
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path.resolve()

    found = conda_python()
    if found is not None:
        return found.resolve()

    return Path(sys.executable).resolve()


def ensure_runtime_python(*modules: str) -> None:
    """Re-exec with the project conda env when IDE/launcher uses wrong python."""
    candidate = runtime_python()
    current = Path(sys.executable).resolve()
    if candidate.is_file() and candidate.resolve() != current:
        os.execv(str(candidate), [str(candidate), *sys.argv])

    required = modules or ("numpy", "torch")
    if all(_can_import(name) for name in required):
        return

    missing = ", ".join(name for name in required if not _can_import(name))
    print(
        "依赖未安装在当前解释器 %s: %s\n"
        "请激活 conda 环境后重试:\n"
        "  conda activate %s\n"
        "或显式指定:\n"
        "  %s <module>/scripts/train.py"
        % (sys.executable, missing, SSM_CONDA_ENV, runtime_python()),
        file=sys.stderr,
    )
    raise SystemExit(1)


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
