from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("SSM_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SsmPaths:
    root: Path
    artifacts_cv: Path
    artifacts_ets: Path
    artifacts_vlm: Path
    data_cv: Path
    data_ets: Path
    configs: Path

    def ensure_runtime_dirs(self) -> None:
        for p in (
            self.artifacts_cv / "runs",
            self.artifacts_cv / "runtime",
            self.artifacts_cv / "engines",
            self.artifacts_cv / "backbones",
            self.artifacts_ets / "runs",
            self.artifacts_ets / "exports",
            self.artifacts_vlm / "models",
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_paths() -> SsmPaths:
    root = repo_root()
    return SsmPaths(
        root=root,
        artifacts_cv=root / "artifacts" / "cv",
        artifacts_ets=root / "artifacts" / "ets",
        artifacts_vlm=root / "artifacts" / "vlm",
        data_cv=root / "data" / "cv",
        data_ets=root / "data" / "ets",
        configs=root / "configs",
    )
