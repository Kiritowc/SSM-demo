import json
import os
from pathlib import Path

import torch
import yaml

from ssm_common.config import load_yaml, merge_dicts
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


task_config = _build_task_config()

taskCfgDir = str(_paths.artifacts_cv / "runs" / "taskCfg")
taskSeqDir = str(_paths.artifacts_cv / "runs" / "tasks")
trainLogDir = str(_paths.artifacts_cv / "runs" / "trainLog")
eventLogDir = str(_paths.artifacts_cv / "runs" / "events")
configDir = str(ROOT / "packages/cv/sunshink_cv/configs")

for d in (taskCfgDir, taskSeqDir, trainLogDir, eventLogDir, configDir):
    os.makedirs(d, exist_ok=True)

TASK_FILE = os.path.join(taskCfgDir, "task_config.json")
TRAIN_LOG = os.path.join(trainLogDir, "train.log")
TASK_HISTORY = os.path.join(trainLogDir, "tasks_history.log")
EVENT_STREAM = os.path.join(eventLogDir, "events.jsonl")
