import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import yaml

from cv.cfg import configDir, taskCfgDir, taskSeqDir


@dataclass(frozen=True)
class TaskDatasetSpec:
    train: str
    val: str
    names: str
    labels: List[str]


@dataclass(frozen=True)
class TaskModelSpec:
    model_name: str
    runs_dir: str
    zerostart: bool = False


@dataclass(frozen=True)
class TaskScheduleSpec:
    allow_train_time_list: List[List[str]]


@dataclass(frozen=True)
class TaskTrainSpec:
    cfg_yaml: Dict


@dataclass(frozen=True)
class TaskTopologySpec:
    schema_version: str
    created_at: str
    dataset: TaskDatasetSpec
    model: TaskModelSpec
    schedule: TaskScheduleSpec
    train: TaskTrainSpec
    extras: Dict = field(default_factory=dict)

    def to_legacy_payload(self) -> Dict:
        payload = {
            "allow_train_time_list": self.schedule.allow_train_time_list,
            "model_name": self.model.model_name,
            "obj_labels": self.dataset.labels,
            "runsDir": self.model.runs_dir,
            "zerostart": self.model.zerostart,
            "cfg_yaml": self.train.cfg_yaml,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }
        payload.update(self.extras)
        return payload


class TaskTopologyCompiler:
    def compile(self, payload: Dict) -> TaskTopologySpec:
        cfg_yaml = payload["cfg_yaml"]
        dataset = TaskDatasetSpec(
            train=cfg_yaml["DATASET"]["TRAIN"],
            val=cfg_yaml["DATASET"]["VAL"],
            names=cfg_yaml["DATASET"]["NAMES"],
            labels=payload["obj_labels"],
        )
        model = TaskModelSpec(
            model_name=payload["model_name"],
            runs_dir=payload["runsDir"],
            zerostart=payload.get("zerostart", False),
        )
        schedule = TaskScheduleSpec(
            allow_train_time_list=payload["allow_train_time_list"]
        )
        extras = {
            key: value
            for key, value in payload.items()
            if key
            not in {"allow_train_time_list", "model_name", "obj_labels", "runsDir", "zerostart", "cfg_yaml"}
        }
        return TaskTopologySpec(
            schema_version="task.topology/v2",
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset=dataset,
            model=model,
            schedule=schedule,
            train=TaskTrainSpec(cfg_yaml=cfg_yaml),
            extras=extras,
        )


class TaskArchiveRepository:
    def enqueue(self, topology: TaskTopologySpec) -> str:
        current_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_file = os.path.join(taskSeqDir, f"{current_ts}.json")
        with open(task_file, "w", encoding="utf-8") as file:
            json.dump(topology.to_legacy_payload(), file, ensure_ascii=False)
        return task_file

    def load(self, path: str) -> TaskTopologySpec:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return TaskTopologyCompiler().compile(payload)


class TaskProjectionResolver:
    def __init__(self, repository=None):
        self.repository = repository or TaskArchiveRepository()

    def resolve(self, source=None) -> TaskTopologySpec:
        if isinstance(source, TaskTopologySpec):
            return source
        if isinstance(source, str):
            return self.repository.load(source)
        default_path = os.path.join(taskCfgDir, "task_config.json")
        if os.path.exists(default_path):
            return self.repository.load(default_path)
        raise FileNotFoundError(f"Task projection source not found: {default_path}")


class TaskConfigMaterializer:
    def materialize(self, topology: TaskTopologySpec):
        cfg_yaml = topology.train.cfg_yaml
        with open(os.path.join(configDir, "self.yaml"), "w", encoding="utf-8") as file:
            yaml.dump(cfg_yaml, file, sort_keys=False, allow_unicode=True)
        with open(os.path.join(configDir, "self.names"), "w", encoding="utf-8") as file:
            for index, label in enumerate(topology.dataset.labels):
                suffix = "" if index == len(topology.dataset.labels) - 1 else "\n"
                file.write(label.strip() + suffix)
        with open(os.path.join(taskCfgDir, "task_config.json"), "w", encoding="utf-8") as file:
            json.dump(topology.to_legacy_payload(), file, ensure_ascii=False)
