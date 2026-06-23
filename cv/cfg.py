"""CV runtime configuration — lazy-initialized from cv/configs/default.yaml."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml

from ssm.config import load_yaml
from ssm.paths import get_paths, repo_root

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_STATE: dict = {}
_initialized = False


def _build_task_config(cfg: dict, paths, root: Path) -> dict:
    runs_rel = cfg.get("runs_dir", "cv/artifacts/runs")
    return {
        "model_name": cfg["model_name"],
        "obj_labels": cfg["obj_labels"],
        "runsDir": str(root / runs_rel),
        "save_dir": str(root / cfg["save_dir"]),
        "zerostart": cfg.get("zerostart", False),
        "pretrained_weight": str(root / cfg["pretrained_weight"]),
        "cfg_yaml": {
            "DATASET": {
                "TRAIN": str(root / cfg["dataset"]["train"]),
                "VAL": str(root / cfg["dataset"]["val"]),
                "NAMES": str(root / cfg["dataset"]["names"]),
            },
            "MODEL": {
                "NC": cfg["model"]["nc"],
                "INPUT_WIDTH": cfg["model"]["input_width"],
                "INPUT_HEIGHT": cfg["model"]["input_height"],
            },
            "TRAIN": {
                "LR": cfg["train"]["lr"],
                "THRESH": cfg["train"]["thresh"],
                "WARMUP": cfg["train"]["warmup"],
                "BATCH_SIZE": cfg["train"]["batch_size"],
                "END_EPOCH": cfg["train"]["end_epoch"],
                "MILESTIONES": cfg["train"]["milestones"],
            },
        },
    }


def _bootstrap_runtime_config(cfg_yaml: dict, labels: list, runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    self_yaml = runtime_dir / "self.yaml"
    self_names = runtime_dir / "self.names"
    with self_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(cfg_yaml, f, sort_keys=False, allow_unicode=True)
    with self_names.open("w", encoding="utf-8") as f:
        f.write("\n".join(label.strip() for label in labels))


def ensure_cv_runtime() -> None:
    """Create artifact dirs and materialize runtime self.yaml / self.names."""
    global _initialized
    if _initialized:
        return

    paths = get_paths()
    paths.ensure_runtime_dirs()
    root = repo_root()
    cv_default = load_yaml(paths.configs_cv / "default.yaml")
    task_config = _build_task_config(cv_default, paths, root)

    runtime_dir = paths.artifacts_cv / "runtime"
    _bootstrap_runtime_config(task_config["cfg_yaml"], task_config["obj_labels"], runtime_dir)

    runs = paths.artifacts_cv / "runs"
    task_cfg_dir = runs / "taskCfg"
    task_seq_dir = runs / "tasks"
    train_log_dir = runs / "trainLog"
    event_log_dir = runs / "events"

    for d in (task_cfg_dir, task_seq_dir, train_log_dir, event_log_dir, runtime_dir):
        os.makedirs(d, exist_ok=True)

    _STATE.update(
        {
            "ROOT": root,
            "task_config": task_config,
            "DEFAULT_RUNS_DIR": str(runs),
            "taskCfgDir": str(task_cfg_dir) + os.sep,
            "taskSeqDir": str(task_seq_dir) + os.sep,
            "trainLogDir": str(train_log_dir) + os.sep,
            "eventLogDir": str(event_log_dir) + os.sep,
            "configDir": str(runtime_dir) + os.sep,
            "TASK_FILE": os.path.join(str(task_cfg_dir), "task_config.json"),
            "TRAIN_LOG": os.path.join(str(train_log_dir), "train.log"),
            "TASK_HISTORY": os.path.join(str(train_log_dir), "tasks_history.log"),
            "EVENT_STREAM": os.path.join(str(event_log_dir), "events.jsonl"),
            "RUNTIME_YAML": str(runtime_dir / "self.yaml"),
            "RUNTIME_NAMES": str(runtime_dir / "self.names"),
        }
    )
    _initialized = True


def __getattr__(name: str):
    if name == "device":
        return device
    ensure_cv_runtime()
    if name in _STATE:
        return _STATE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "device",
    "ensure_cv_runtime",
    "task_config",
    "DEFAULT_RUNS_DIR",
    "taskCfgDir",
    "taskSeqDir",
    "trainLogDir",
    "configDir",
    "TASK_FILE",
    "TRAIN_LOG",
    "TASK_HISTORY",
    "EVENT_STREAM",
    "RUNTIME_YAML",
    "RUNTIME_NAMES",
]
