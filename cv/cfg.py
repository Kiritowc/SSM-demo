import json
import os
from pathlib import Path

import torch
import yaml

from ssm_common.config import load_yaml
from ssm_common.paths import get_paths, repo_root


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_paths = get_paths()
_paths.ensure_runtime_dirs()

ROOT = repo_root()
CV_DEFAULT = load_yaml(_paths.configs / "cv" / "default.yaml")


def _build_task_config() -> dict:
    cfg = CV_DEFAULT
    return {
        "allow_train_time_list": cfg.get("allow_train_time_list", []),
        "model_name": cfg["model_name"],
        "obj_labels": cfg["obj_labels"],
        "runsDir": str(_paths.artifacts_cv / "runs"),
        "save_dir": str(ROOT / cfg["save_dir"]),
        "zerostart": cfg.get("zerostart", False),
        "pretrained_weight": str(ROOT / cfg["pretrained_weight"]),
        "cfg_yaml": {
            "DATASET": {
                "TRAIN": str(ROOT / cfg["dataset"]["train"]),
                "VAL": str(ROOT / cfg["dataset"]["val"]),
                "NAMES": str(ROOT / cfg["dataset"]["names"]),
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


def _bootstrap_runtime_config(cfg_yaml: dict, labels: list) -> None:
    """Materialize runtime YAML consumed by LoadYaml and training."""
    runtime_dir = _paths.artifacts_cv / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    self_yaml = runtime_dir / "self.yaml"
    self_names = runtime_dir / "self.names"
    with self_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(cfg_yaml, f, sort_keys=False, allow_unicode=True)
    with self_names.open("w", encoding="utf-8") as f:
        f.write("\n".join(label.strip() for label in labels))


task_config = _build_task_config()
_bootstrap_runtime_config(task_config["cfg_yaml"], task_config["obj_labels"])

taskCfgDir = str(_paths.artifacts_cv / "runs" / "taskCfg")
taskSeqDir = str(_paths.artifacts_cv / "runs" / "tasks")
trainLogDir = str(_paths.artifacts_cv / "runs" / "trainLog")
eventLogDir = str(_paths.artifacts_cv / "runs" / "events")
configDir = str(_paths.artifacts_cv / "runtime")

for d in (taskCfgDir, taskSeqDir, trainLogDir, eventLogDir, configDir):
    os.makedirs(d, exist_ok=True)

TASK_FILE = os.path.join(taskCfgDir, "task_config.json")
TRAIN_LOG = os.path.join(trainLogDir, "train.log")
TASK_HISTORY = os.path.join(trainLogDir, "tasks_history.log")
EVENT_STREAM = os.path.join(eventLogDir, "events.jsonl")
RUNTIME_YAML = os.path.join(configDir, "self.yaml")
RUNTIME_NAMES = os.path.join(configDir, "self.names")
