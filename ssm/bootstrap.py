"""Ensure SSM_ROOT is set and repo root is importable (standalone scripts)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


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


def ensure_runtime_python(*modules: str) -> None:
    """Re-exec with active conda env when IDE/launcher uses system python."""
    required = modules or ("numpy", "torch")
    if all(_can_import(name) for name in required):
        return

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidate = Path(prefix) / "bin" / "python"
        if candidate.is_file() and candidate.resolve() != Path(sys.executable).resolve():
            os.execv(str(candidate), [str(candidate), *sys.argv])

    print(
        "依赖未安装在当前解释器 %s: %s\n"
        "请使用 conda 环境运行，例如:\n"
        "  python <module>/scripts/train.py\n"
        "  ~/miniconda3/envs/ssdet/bin/python <module>/scripts/train.py"
        % (sys.executable, ", ".join(name for name in required if not _can_import(name))),
        file=sys.stderr,
    )
    raise SystemExit(1)


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
