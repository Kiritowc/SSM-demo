import copy
import os

import yaml

from sunshink_cv.cfg import EVENT_STREAM, device, task_config, trainLogDir
from sunshink_cv.core.assembly import DetectorPlatformAssembly
from sunshink_cv.core.backbone import MoEn
from sunshink_cv.core.events import EventBus, JsonlEventSink
from sunshink_cv.core.runtime import TrainingRuntimeContext
from sunshink_cv.core.training import DetectorTrainingOrchestrator, TaskStatusLedger


class sunshinkDet:
    def __init__(
        self,
        yaml="cv/configs/self.yaml",
        weight=None,
        model=None,
        ins=None,
        ous=None,
        aug=True,
        method=1,
        debug=False,
        epochs=9,
        delta=10,
        spp="spp",
        dir="runs/",
        save_dir=None,
    ):
        self.yaml = yaml
        self.weight = weight
        self.model = model
        self.ins = ins
        self.ous = ous
        self.aug = aug
        self.method = method
        self.debug = debug
        self.epochs = epochs
        self.delta = delta
        self.spp = spp
        self.dir = dir
        self.model_name = model
        self.save_dir_override = save_dir
        assert os.path.exists(self.yaml), "请指定正确的配置文件路径"

        self.pid = os.getpid()
        self.status_ledger = TaskStatusLedger()
        self._persist_process_identity()
        self.saveDir = self._resolve_save_dir()
        self.vault = MoEn()

        self.runtime = TrainingRuntimeContext.from_legacy_opt(
            self, self.yaml, device, self.saveDir
        )
        self.cfg = self.runtime.cfg
        self.train_log = self.runtime.paths.train_log_path
        self.event_bus = EventBus(
            [
                JsonlEventSink(EVENT_STREAM),
                JsonlEventSink(os.path.join(self.saveDir, "telemetry.jsonl")),
            ]
        )

        bundle = DetectorPlatformAssembly().build(self.runtime)
        self.model = bundle.model
        self.optimizer = bundle.optimizer
        self.scheduler = bundle.scheduler
        self.ema = bundle.ema
        self.loss_function = bundle.loss_function
        self.evaluation = bundle.evaluation
        self.train_dataloader = bundle.train_dataloader
        self.val_dataloader = bundle.val_dataloader

        self._bootstrap_task_ledger()
        self.event_bus.emit(
            "training",
            "initialized",
            {
                "model_name": self.model_name,
                "save_dir": self.saveDir,
                "yaml": self.yaml,
                "epochs": self.epochs,
                "batch_size": self.cfg.batch_size,
            },
        )
        self.orchestrator = DetectorTrainingOrchestrator(self)

    def _persist_process_identity(self):
        print(f"Current Process PID: {self.pid}")
        with open(os.path.join(trainLogDir, "task_pid.txt"), "w", encoding="utf-8") as file:
            file.write(str(self.pid))

    def _resolve_save_dir(self):
        if self.save_dir_override:
            save_dir = os.path.abspath(os.path.expanduser(str(self.save_dir_override)))
            os.makedirs(save_dir, exist_ok=True)
            return save_dir if save_dir.endswith(os.sep) else save_dir + os.sep
        save_dir = os.path.join(self.dir, self.model_name)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir if save_dir.endswith(os.sep) else save_dir + os.sep

    def _bootstrap_task_ledger(self):
        try:
            payload = self.status_ledger.load()
            if not payload:
                payload = copy.deepcopy(task_config)
                payload["model_name"] = self.model_name
                payload["runsDir"] = self.dir
                with open(self.cfg.names, "r", encoding="utf-8") as file:
                    payload["obj_labels"] = [
                        one.strip() for one in file.readlines() if one.strip()
                    ]
                with open(self.yaml, "r", encoding="utf-8") as file:
                    payload["cfg_yaml"] = yaml.safe_load(file)
            payload["pid"] = str(self.pid)
            payload["status"] = "training"
            self.status_ledger.patch(**payload)
        except Exception as exc:
            print("读取训练任务PID修改训练任务配置出错: ", exc)

    def train(self):
        return self.orchestrator.run()
