"""Checkpoint save/load utilities."""

from __future__ import annotationsfrom pathlib import Pathfrom typing import Any, Literalimport joblibimport torchWEIGHTS_DIRNAME = "weights"


def weights_dir(run_dir: str | Path) -> Path:
    """Return the weights subdirectory for a training run."""
    return Path(run_dir) / WEIGHTS_DIRNAME


def best_checkpoint_path(run_dir: str | Path) -> Path:
    """Path to best model checkpoint for a run."""
    return weights_dir(run_dir) / "best.pt"


def last_checkpoint_path(run_dir: str | Path) -> Path:
    """Path to last epoch checkpoint for a run."""
    return weights_dir(run_dir) / "last.pt"


def run_dir_from_checkpoint(checkpoint_path: str | Path) -> Path:
    """Resolve experiment run root from a checkpoint file path."""
    path = Path(checkpoint_path).resolve()
    if path.parent.name == WEIGHTS_DIRNAME:
        return path.parent.parent
    return path.parent


def resolve_checkpoint(
    path: str | Path,
    prefer: Literal["best", "last"] = "best",
) -> Path:
    """
    Resolve a checkpoint path from an explicit file or run directory.

    Supports:
    - ``runs/exp/weights/best.pt``
    - ``runs/exp/weights/``
    - ``runs/exp/`` (auto-picks ``weights/best.pt``)
    - legacy flat layout ``runs/exp/best.pt``
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {candidate}")

    filename = f"{prefer}.pt"
    search_paths = [
        candidate / WEIGHTS_DIRNAME / filename,
        candidate / filename,
    ]
    if candidate.name == WEIGHTS_DIRNAME:
        search_paths.insert(0, candidate / filename)

    for item in search_paths:
        if item.is_file():
            return item.resolve()

    raise FileNotFoundError(
        f"No checkpoint '{filename}' found under {candidate}. "
        f"Expected {WEIGHTS_DIRNAME}/{filename} or legacy {filename}."
    )


def _find_artifact(checkpoint_path: Path, filename: str) -> Path | None:
    """Find scaler artifact next to checkpoint, with legacy layout fallback."""
    parent = checkpoint_path.parent
    candidates = [parent / filename]
    if parent.name == WEIGHTS_DIRNAME:
        candidates.append(parent.parent / filename)
    else:
        candidates.append(parent / WEIGHTS_DIRNAME / filename)
    for path in candidates:
        if path.exists():
            return path
    return None


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    cfg: dict[str, Any],
    scaler: Any | None = None,
    target_scaler: Any | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint with model, optimizer, and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "cfg": cfg,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        state.update(extra)

    torch.save(state, path)

    if scaler is not None:
        feature_scaler_path = path.parent / "scaler.joblib"
        joblib.dump(scaler, feature_scaler_path)

    if target_scaler is not None:
        target_scaler_path = path.parent / "target_scaler.joblib"
        joblib.dump(target_scaler, target_scaler_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load checkpoint into model and optionally optimizer."""
    path = Path(path)
    map_location = device if device is not None else "cpu"
    state = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return state


def load_scaler(checkpoint_path: str | Path) -> Any | None:
    """Load feature scaler saved alongside checkpoint."""
    artifact = _find_artifact(Path(checkpoint_path), "scaler.joblib")
    if artifact is not None:
        return joblib.load(artifact)
    return None


def load_target_scaler(checkpoint_path: str | Path) -> Any | None:
    """Load target scaler saved alongside checkpoint."""
    artifact = _find_artifact(Path(checkpoint_path), "target_scaler.joblib")
    if artifact is not None:
        return joblib.load(artifact)
    return None
