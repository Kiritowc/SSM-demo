#!/usr/bin/env python
"""SSDet training entry point."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo, ensure_runtime_python

REPO_ROOT = bootstrap_repo(_REPO)
ensure_runtime_python("numpy", "torch")

from ssdet.cfg import RUNTIME_YAML, taskCfgDir, task_config
from ssdet.core.tasking import TaskConfigMaterializer, TaskTopologyCompiler


def ensure_task_config(configfile: str) -> None:
    """Seed task_config.json from ssdet/configs/default.yaml when missing."""
    if os.path.exists(configfile):
        return

    configs = copy.deepcopy(task_config)
    os.makedirs(os.path.dirname(configfile), exist_ok=True)
    with open(configfile, "w", encoding="utf-8") as file:
        json.dump(configs, file, ensure_ascii=False, indent=2)
    print(f"已根据 ssdet/configs/default.yaml 生成: {configfile}")


def load_task_config(configfile: str) -> dict:
    with open(configfile, "r", encoding="utf-8") as file:
        return json.load(file)


class TrainRunner:
    def __init__(self, configfile: str) -> None:
        self.configfile = configfile

    @staticmethod
    def _next_archive_dir(runs_dir: str, model_name: str) -> str:
        base_name = "%s_%s" % (model_name, datetime.now().strftime("%Y-%m-%d"))
        candidate = os.path.join(runs_dir, base_name)
        index = 1
        while os.path.exists(candidate):
            candidate = os.path.join(runs_dir, "%s_%d" % (base_name, index))
            index += 1
        return candidate

    def launch(self) -> dict:
        ensure_task_config(self.configfile)
        configs = load_task_config(self.configfile)
        self._materialize_task_config(configs)

        from ssdet.core.training import DetectorTrainer

        model_name = configs["model_name"]
        epochs = configs["cfg_yaml"]["TRAIN"]["END_EPOCH"]
        runs_dir = configs["runsDir"]
        if configs.get("zerostart"):
            save_ov = configs.get("save_dir")
            if save_ov:
                model_dir = os.path.abspath(os.path.expanduser(str(save_ov)))
                archive_root = os.path.dirname(model_dir.rstrip(os.sep)) or runs_dir
                tag = os.path.basename(model_dir.rstrip(os.sep)) or model_name
            else:
                model_dir = os.path.join(runs_dir, model_name)
                archive_root = runs_dir
                tag = model_name
            if os.path.exists(model_dir):
                shutil.move(model_dir, self._next_archive_dir(archive_root, tag))
        weight = configs.get("pretrained_weight")
        if weight and not os.path.exists(weight):
            print("预训练权重不存在，从头训练: %s" % weight)
            weight = None
        model = DetectorTrainer(
            yaml=RUNTIME_YAML,
            model=model_name,
            weight=weight,
            epochs=epochs,
            dir=runs_dir,
            save_dir=configs.get("save_dir"),
        )
        model.train()
        print("训练任务完成!")
        gc.collect()
        return {"model_name": model_name, "runs_dir": runs_dir}

    @staticmethod
    def _materialize_task_config(configs: dict) -> None:
        topology = TaskTopologyCompiler().compile(configs)
        TaskConfigMaterializer().materialize(topology)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--configfile", type=str, default=os.path.join(taskCfgDir, "task_config.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TrainRunner(args.configfile).launch()


if __name__ == "__main__":
    main()
