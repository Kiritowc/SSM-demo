import argparse
import copy
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime

import yaml
from tqdm import tqdm

from ssdet.cfg import EVENT_STREAM, device, task_config, taskCfgDir, trainLogDir
from ssdet.core.assembly import DetectorPlatformAssembly
from ssdet.core.backbone.mi import MoEn
from ssdet.core.events import EventBus, JsonlEventSink
from ssdet.core.runtime import TrainingRuntimeContext
from ssdet.paths import RUNS_DIR, RUNTIME_YAML
from ssdet.core.registry import NamedComponentRegistry
from ssdet.core.exporting import convert_and_save_model
from ssdet.utils.tool import load_training_state, save_training_state


warmup_policy_registry = NamedComponentRegistry("warmup-policy")
checkpoint_policy_registry = NamedComponentRegistry("checkpoint-policy")
time_window_policy_registry = NamedComponentRegistry("time-window-policy")
finalizer_policy_registry = NamedComponentRegistry("finalizer-policy")


@dataclass
class TrainingCursor:
    total_epoch: int
    start_epoch: int = 0
    epoch: int = 0
    batch_num: int = 0
    best_map05: float = 0.0
    last_iou: float = 0.0
    last_loss: float = 0.0


@dataclass(frozen=True)
class EpochTelemetry:
    epoch: int
    batch_num: int
    iou: float
    loss: float


@dataclass(frozen=True)
class EvaluationSnapshot:
    epoch: int
    metrics: dict


class TaskStatusLedger:
    def __init__(self, task_config_path=None):
        self.task_config_path = task_config_path or os.path.join(
            taskCfgDir, "task_config.json"
        )

    def load(self):
        if not os.path.exists(self.task_config_path):
            return {}
        with open(self.task_config_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def patch(self, **fields):
        payload = self.load()
        payload.update(fields)
        with open(self.task_config_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)
        return payload


class PolynomialWarmupRegulator:
    def __init__(self, runtime, dataloader):
        self.learn_rate = runtime.cfg.learn_rate
        self.warmup_num = max(1, 5 * len(dataloader))

    def apply(self, optimizer, batch_num):
        for group in optimizer.param_groups:
            if batch_num <= self.warmup_num:
                scale = math.pow(batch_num / self.warmup_num, 4)
                group["lr"] = self.learn_rate * scale
        return optimizer.param_groups[-1]["lr"]


class EvaluationCheckpointRelay:
    def __init__(self, trainer):
        self.trainer = trainer
        self.vault = trainer.vault

    def evaluate_and_commit(self, cursor: TrainingCursor):
        trainer = self.trainer
        trainer.model.eval()
        print("computer mAP...")
        print("epoch: ", cursor.epoch)
        metrics = trainer.evaluation.compute_map(trainer.val_dataloader, trainer.model)
        print("mAP0.5:", metrics["mAP0.5"])
        print("mAP0.5:0.95:", metrics["mAP0.5:0.95"])
        print("Precision:", metrics["precision"])
        print("Recall:", metrics["recall"])
        print("F1:", metrics["F1"])
        trainer.event_bus.emit(
            "evaluation",
            "completed",
            {
                "model_name": trainer.model_name,
                "epoch": cursor.epoch,
                "metrics": metrics,
            },
        )
        self._persist_latest_artifacts()
        if metrics["mAP0.5"] > cursor.best_map05:
            cursor.best_map05 = metrics["mAP0.5"]
            self._persist_best_artifacts()
        return EvaluationSnapshot(epoch=cursor.epoch, metrics=metrics)

    def _persist_latest_artifacts(self):
        trainer = self.trainer
        last_model_path = os.path.join(trainer.saveDir, "last.bin")
        ema_last_model_path = os.path.join(trainer.saveDir, "ema_last.bin")
        self.vault.en_save_model(trainer.model.state_dict(), last_model_path)
        self.vault.en_save_model(trainer.ema.shadow, ema_last_model_path)

    def _persist_best_artifacts(self):
        trainer = self.trainer
        last_model_path = os.path.join(trainer.saveDir, "last.bin")
        best_model_path = os.path.join(trainer.saveDir, "best.bin")
        ema_best_model_path = os.path.join(trainer.saveDir, "ema_best.bin")
        print("当前最优mAP0.5更新为更高值，正在固化最佳资产......")
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        shutil.copy(last_model_path, best_model_path)
        self.vault.en_save_model(trainer.ema.shadow, ema_best_model_path)


class TemporalSleepPolicy:
    def __init__(self, trainer):
        self.trainer = trainer

    def maybe_suspend(self, cursor: TrainingCursor):
        return cursor


class EncryptedArtifactFinalizer:
    def __init__(self, trainer):
        self.trainer = trainer

    def finalize(self, cursor: TrainingCursor):
        trainer = self.trainer
        with open(trainer.cfg.val_txt, encoding="utf-8") as file:
            val_list = [one.strip() for one in file.readlines() if one.strip()]
        opt = argparse.Namespace(
            yaml=trainer.yaml,
            model=trainer.model_name,
            weight=os.path.join(trainer.saveDir, "best.bin"),
            save_path=os.path.join(trainer.saveDir, "run.bin"),
            img=random.choice(val_list),
            thresh=0.50,
            spp="spp",
            ins=None,
            ous=None,
        )
        convert_and_save_model(opt)
        print("当前模型已经成功转化存储完成......")
        trainer.event_bus.emit(
            "export",
            "completed",
            {
                "model_name": trainer.model_name,
                "save_path": os.path.join(trainer.saveDir, "run.bin"),
                "terminal_epoch": cursor.epoch,
            },
        )


class DetectorTrainingOrchestrator:
    def __init__(self, trainer):
        self.trainer = trainer
        self.warmup_policy = warmup_policy_registry.build(
            trainer.runtime.dialect.warmup, trainer.runtime, trainer.train_dataloader
        )
        self.checkpoint_policy = checkpoint_policy_registry.build(
            trainer.runtime.dialect.checkpoint, trainer
        )
        self.time_window_policy = time_window_policy_registry.build(
            trainer.runtime.dialect.time_window, trainer
        )
        self.finalizer_policy = finalizer_policy_registry.build(
            trainer.runtime.dialect.finalizer, trainer
        )

    def run(self):
        cursor = self._bootstrap_cursor()
        trainer = self.trainer
        trainer.event_bus.emit(
            "training",
            "started",
            {
                "model_name": trainer.model_name,
                "start_epoch": cursor.start_epoch,
                "total_epoch": cursor.total_epoch,
                "batch_num": cursor.batch_num,
            },
        )
        for epoch in range(cursor.start_epoch, cursor.total_epoch):
            cursor.epoch = epoch
            self._run_epoch(cursor)
            self._flush_epoch_telemetry(cursor)
            if epoch % int(trainer.delta) == 0:
                self.checkpoint_policy.evaluate_and_commit(cursor)
            save_training_state(
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                epoch + 1,
                cursor.batch_num,
                trainer.ema,
                trainer.saveDir,
            )
            cursor = self.time_window_policy.maybe_suspend(cursor)
            print("[-]" * 30)
            trainer.scheduler.step()
        self._finalize(cursor)
        return cursor

    def _bootstrap_cursor(self):
        trainer = self.trainer
        total_epoch = trainer.epochs if trainer.epochs else trainer.cfg.end_epoch
        print("Starting training for %g epochs..." % total_epoch)
        if not os.path.exists(trainer.train_log):
            with open(trainer.train_log, "w", encoding="utf-8") as file:
                file.write("Epoch,IOU,Loss\n")
        trainer.ema.register()
        (
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.ema,
            start_epoch,
            batch_num,
        ) = load_training_state(
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.ema,
            trainer.saveDir,
        )
        if start_epoch >= total_epoch:
            print(
                "Checkpoint epoch %d >= target %d; discarding checkpoint, training from epoch 0"
                % (start_epoch, total_epoch)
            )
            state_path = os.path.join(trainer.saveDir.rstrip(os.sep), "training_state.bin")
            if os.path.exists(state_path):
                os.remove(state_path)
            bundle = DetectorPlatformAssembly().build(trainer.runtime)
            trainer.model = bundle.model
            trainer.optimizer = bundle.optimizer
            trainer.scheduler = bundle.scheduler
            trainer.ema = bundle.ema
            trainer.ema.register()
            start_epoch = 0
            batch_num = 0
        if start_epoch == 0:
            print(f"Training from epoch {start_epoch} to epoch {total_epoch}")
        else:
            print(f"Resuming training from epoch {start_epoch} to epoch {total_epoch}")
        print("Now batch_num is: ", batch_num)
        return TrainingCursor(
            total_epoch=total_epoch,
            start_epoch=start_epoch,
            batch_num=batch_num,
        )

    def _run_epoch(self, cursor: TrainingCursor):
        trainer = self.trainer
        print("epoch: ", cursor.epoch)
        trainer.model.train()
        pbar = tqdm(trainer.train_dataloader)
        info = "Epoch:%d IOU:%f Loss:%f" % (
            cursor.epoch,
            cursor.last_iou,
            cursor.last_loss,
        )
        for imgs, targets in pbar:
            imgs = imgs.to(trainer.runtime.device).float() / 255.0
            targets = targets.to(trainer.runtime.device)
            preds = trainer.model(imgs)
            iou, obj, cls, total = trainer.loss_function(preds, targets)
            total.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.ema.update()
            self.warmup_policy.apply(trainer.optimizer, cursor.batch_num)
            info = "Epoch:%d IOU:%f Loss:%f" % (cursor.epoch, iou, total)
            pbar.set_description(info)
            cursor.last_iou = float(iou.detach().item())
            cursor.last_loss = float(total.detach().item())
            with open(trainer.train_log, "a", encoding="utf-8") as file:
                file.write(f"{cursor.epoch},{iou:.6f},{total:.6f}\n")
            cursor.batch_num += 1
        with open(trainer.train_log, "a", encoding="utf-8") as file:
            file.write(info + "\n")

    def _flush_epoch_telemetry(self, cursor: TrainingCursor):
        trainer = self.trainer
        epoch_telemetry = EpochTelemetry(
            epoch=cursor.epoch,
            batch_num=cursor.batch_num,
            iou=cursor.last_iou,
            loss=cursor.last_loss,
        )
        trainer.event_bus.emit(
            "training",
            "epoch-completed",
            {
                "model_name": trainer.model_name,
                "epoch": epoch_telemetry.epoch,
                "batch_num": epoch_telemetry.batch_num,
                "iou": epoch_telemetry.iou,
                "loss": epoch_telemetry.loss,
            },
        )

    def _finalize(self, cursor: TrainingCursor):
        trainer = self.trainer
        try:
            self.finalizer_policy.finalize(cursor)
        except Exception as exc:
            print("模型转化存储出现异常问题: ", exc)
            trainer.event_bus.emit(
                "export",
                "failed",
                {
                    "model_name": trainer.model_name,
                    "error": str(exc),
                },
            )
        trainer.status_ledger.patch(status="finished")
        print("当前训练任务已完成，任务状态已修改......")
        shutil.copy(
            os.path.join(taskCfgDir, "task_config.json"),
            os.path.join(trainer.saveDir, "task_config.json"),
        )
        trainer.event_bus.emit(
            "training",
            "finished",
            {
                "model_name": trainer.model_name,
                "status": "finished",
                "save_dir": trainer.saveDir,
            },
        )


warmup_policy_registry.register("polynomial-warmup", PolynomialWarmupRegulator)
checkpoint_policy_registry.register("evaluation-checkpoint", EvaluationCheckpointRelay)
time_window_policy_registry.register("temporal-sleep", TemporalSleepPolicy)
finalizer_policy_registry.register("encrypted-artifact-finalizer", EncryptedArtifactFinalizer)


class DetectorTrainer:
    """PyTorch detector training entry — ONNX runtime uses ssdet.inference.ssDet."""

    def __init__(
        self,
        yaml=RUNTIME_YAML,
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
        dir=RUNS_DIR,
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
