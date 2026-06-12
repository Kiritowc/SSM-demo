import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import List

from cv.cfg import *
from cv.core.events import EventBus, JsonlEventSink
from cv.core.tasking import TaskArchiveRepository, TaskConfigMaterializer, TaskProjectionResolver


class TaskConfigProjection:
    def __init__(self):
        self.materializer = TaskConfigMaterializer()
        self.resolver = TaskProjectionResolver()

    def materialize(self, task_config=None):
        topology = self.resolver.resolve(task_config)
        self.materializer.materialize(topology)
        print("配置文件自动创建生成完成!")


class TrainingSubprocessCapsule:
    def __init__(self):
        self.events = EventBus([JsonlEventSink(EVENT_STREAM)])

    def launch(self, configfile):
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
        self.events.emit("task", "launch", {"model_name": model_name, "configfile": configfile, "runs_dir": runs_dir})

        popen_kwargs = {
            "args": ["python", "-m", "cv.train", "--configfile", configfile],
            "stdout": None,
            "stderr": None,
        }
        with open(TRAIN_LOG, "w") as log_file:
            popen_kwargs["stdout"] = log_file
            popen_kwargs["stderr"] = log_file
            if sys.platform.startswith("win"):
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            process = subprocess.Popen(**popen_kwargs)
            print(f"Training task started with PID: {process.pid}")
            with open(taskCfgDir + "task_pid.txt", "w") as file:
                file.write(str(process.pid))
            process.wait()

        gc.collect()
        if process.returncode != 0:
            print(f"Training task failed with return code: {process.returncode}")
            self.events.emit("task", "failed", {"model_name": model_name, "returncode": process.returncode})
        else:
            print("Training task completed successfully")
            self.events.emit("task", "completed", {"model_name": model_name, "returncode": process.returncode})

        os.remove(configfile)
        with open(TASK_HISTORY, "a") as file:
            file.write("结束时间: " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "\n")
            file.write("[-]" * 60 + "\n")


class TaskQueueClockwork:
    def __init__(self):
        self.config_projection = TaskConfigProjection()
        self.capsule = TrainingSubprocessCapsule()
        self.repository = TaskArchiveRepository()
        self.events = EventBus([JsonlEventSink(EVENT_STREAM)])

    def pulse(self):
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Current time: {current_time}")
            task_files = [
                taskSeqDir + filename
                for filename in os.listdir(taskSeqDir)
                if filename.endswith(".json")
            ]
            if task_files:
                task_files.sort()
                task_file = task_files[0]
                print("开始执行训练任务: ", task_file)
                topology = self.repository.load(task_file)
                self.events.emit("queue", "dequeue", {"task_file": task_file, "model_name": topology.model.model_name})
                self.config_projection.materialize(task_file)
                print(f"Starting training task from {task_file}")
                self.capsule.launch(task_file)
            else:
                print("No pending training tasks. Waiting for new tasks...")
                time.sleep(60)


class GuardianThreadMatrix:
    def __init__(self):
        self.clockwork = TaskQueueClockwork()

    def bootstrap(self):
        onclock_thread = threading.Thread(target=self.clockwork.pulse)
        onclock_thread.setName("Thread:onclock")
        onclock_thread.start()

        init_threads = threading.enumerate()
        init_thread_names = [thread.getName() for thread in init_threads]
        check_thread = threading.Thread(
            target=self.watch_forever, args=(1800, init_thread_names)
        )
        check_thread.setName("Thread:check")
        check_thread.start()

    def watch_forever(self, sleep_time=60, init_threads_names: List[str] = None):
        init_threads_names = init_threads_names or []
        while True:
            current_threads = threading.enumerate()
            now_threads_names = [thread.getName() for thread in current_threads]
            print("当前运行线程:{0}".format(now_threads_names))
            for thread_name in init_threads_names:
                if thread_name in now_threads_names:
                    continue
                print("===" + thread_name + "stopped,now restart")
                if thread_name == "Thread:onclock":
                    onclock_thread = threading.Thread(target=self.clockwork.pulse)
                    onclock_thread.setName(thread_name)
                    onclock_thread.start()
            time.sleep(sleep_time)
