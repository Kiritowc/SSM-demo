import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

from cv.cfg import EVENT_STREAM, TASK_HISTORY, TRAIN_LOG, taskSeqDir, trainLogDir
from cv.core.events import EventBus, JsonlEventSink
from cv.core.tasking import TaskArchiveRepository, TaskConfigMaterializer, TaskProjectionResolver


class TrainProcess:
    def __init__(self):
        self.events = EventBus([JsonlEventSink(EVENT_STREAM)])

    def run(self, configfile: str) -> None:
        with open(configfile) as file:
            configs = json.load(file)
        model_name = configs["model_name"]
        runs_dir = configs["runsDir"]
        if configs["zerostart"]:
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
                shutil.move(
                    model_dir,
                    os.path.join(
                        archive_root, f"{tag}_{datetime.now().strftime('%Y-%m-%d')}"
                    ),
                )

        with open(TASK_HISTORY, "a") as file:
            file.write("模型名称: " + str(model_name) + "\n")
            file.write("任务名称: " + str(configfile) + "\n")
            file.write("开始时间: " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "\n")
        self.events.emit("task", "launch", {"model_name": model_name, "configfile": configfile})

        from ssm.paths import repo_root

        train_script = str(repo_root() / "cv" / "scripts" / "train.py")
        with open(TRAIN_LOG, "w") as log_file:
            process = subprocess.Popen(
                [sys.executable, train_script, "--configfile", configfile],
                stdout=log_file,
                stderr=log_file,
            )
            with open(os.path.join(trainLogDir, "task_pid.txt"), "w") as pid_file:
                pid_file.write(str(process.pid))
            process.wait()

        pid_file = os.path.join(trainLogDir, "task_pid.txt")
        if os.path.exists(pid_file):
            os.remove(pid_file)

        gc.collect()
        status = "completed" if process.returncode == 0 else "failed"
        print("Training task %s (rc=%s)" % (status, process.returncode))
        self.events.emit("task", status, {"model_name": model_name, "returncode": process.returncode})

        if process.returncode == 0:
            self._deploy_after_train(configs)

        os.remove(configfile)
        with open(TASK_HISTORY, "a") as file:
            file.write("结束时间: " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "\n")
            file.write("[-]" * 60 + "\n")

    def _deploy_after_train(self, configs: dict) -> None:
        from ssm.paths import repo_root

        save_dir = configs.get("save_dir") or os.path.join(
            configs["runsDir"], configs["model_name"]
        )
        val_txt = configs["cfg_yaml"]["DATASET"]["VAL"]
        deploy_script = repo_root() / "cv" / "scripts" / "deploy.py"
        cmd = [
            sys.executable,
            str(deploy_script),
            "--save-dir",
            str(save_dir),
            "--model-name",
            str(configs["model_name"]),
            "--val-txt",
            str(val_txt),
            "--restart-camera",
        ]
        print("+ deploy after train:", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=str(repo_root()))
        deploy_status = "completed" if result.returncode == 0 else "failed"
        print("Deploy task %s (rc=%s)" % (deploy_status, result.returncode), flush=True)
        self.events.emit(
            "deploy",
            deploy_status,
            {"model_name": configs["model_name"], "returncode": result.returncode},
        )


class TrainQueue:
    def __init__(self):
        self.process = TrainProcess()
        self.repository = TaskArchiveRepository()
        self.materializer = TaskConfigMaterializer()
        self.resolver = TaskProjectionResolver()
        self.events = EventBus([JsonlEventSink(EVENT_STREAM)])

    def loop(self) -> None:
        while True:
            task_files = sorted(
                taskSeqDir + name
                for name in os.listdir(taskSeqDir)
                if name.endswith(".json")
            )
            if not task_files:
                time.sleep(60)
                continue

            task_file = task_files[0]
            topology = self.repository.load(task_file)
            self.events.emit(
                "queue",
                "dequeue",
                {"task_file": task_file, "model_name": topology.model.model_name},
            )
            self.materializer.materialize(self.resolver.resolve(task_file))
            self.process.run(task_file)


def start_train_queue() -> None:
    thread = threading.Thread(target=TrainQueue().loop, name="train-queue", daemon=True)
    thread.start()
