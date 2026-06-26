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
    ssdet: Path
    ets: Path
    vlm: Path
    apps: Path
    apps_web: Path
    apps_video: Path
    configs_ssdet: Path
    configs_ets: Path
    configs_vlm: Path
    configs_platform: Path
    artifacts_ssdet: Path
    artifacts_ets: Path
    artifacts_vlm: Path
    data_ssdet: Path
    data_ets: Path

    def ensure_runtime_dirs(self) -> None:
        for p in (
            self.artifacts_ssdet / "runs",
            self.artifacts_ssdet / "runtime",
            self.artifacts_ssdet / "engines",
            self.artifacts_ssdet / "backbones",
            self.artifacts_ssdet / "logs",
            self.artifacts_ets / "runs",
            self.artifacts_ets / "exports",
            self.artifacts_vlm / "models",
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_paths() -> SsmPaths:
    root = repo_root()
    ssdet_dir = root / "ssdet"
    ets = root / "ets"
    vlm = root / "vlm"
    apps = root / "apps"
    return SsmPaths(
        root=root,
        ssdet=ssdet_dir,
        ets=ets,
        vlm=vlm,
        apps=apps,
        apps_web=apps / "web",
        apps_video=apps / "video",
        configs_ssdet=ssdet_dir / "configs",
        configs_ets=ets / "configs",
        configs_vlm=vlm / "configs",
        configs_platform=root / "configs",
        artifacts_ssdet=ssdet_dir / "artifacts",
        artifacts_ets=ets / "artifacts",
        artifacts_vlm=vlm / "artifacts",
        data_ssdet=ssdet_dir / "data",
        data_ets=ets / "datasets",
    )
