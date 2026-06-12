import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime

from tqdm import tqdm

from cv.cfg import taskCfgDir
from cv.core.registry import NamedComponentRegistry
from cv.export import convert_and_save_model
from cv.utils.tool import (
    calculate_next_training_time,
    is_training_time,
    load_training_state,
    save_training_state,
)


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
        self.ledger = trainer.status_ledger

    def maybe_suspend(self, cursor: TrainingCursor):
        if is_training_time():
            return cursor
        trainer = self.trainer
        print("超出允许训练的时间段,停止训练,存储当前训练状态信息......")
        save_training_state(
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            cursor.epoch + 1,
            cursor.batch_num,
            trainer.ema,
            trainer.saveDir,
        )
        trainer.event_bus.emit(
            "training",
            "sleep-enter",
            {
                "model_name": trainer.model_name,
                "epoch": cursor.epoch,
                "batch_num": cursor.batch_num,
            },
        )
        self.ledger.patch(status="sleeping")
        print("已经停止训练，任务状态已修改......")
        allow_train_time_list = self.ledger.load().get(
            "allow_train_time_list", [["01:01:01", "23:59:59"]]
        )
        next_start_time = calculate_next_training_time(allow_train_time_list)
        print("下次自动执行起点时间: ", next_start_time)
        next_start_time = datetime.strptime(next_start_time, "%Y-%m-%d %H:%M:%S")
        time_until_next = (next_start_time - datetime.now()).total_seconds()
        print(
            f"Next training will start at {next_start_time.strftime('%Y-%m-%d %H:%M:%S')}. Waiting..."
        )
        print("执行自动休眠操作,休眠时长: ", time_until_next)
        time.sleep(max(time_until_next, 0))
        trainer.event_bus.emit(
            "training",
            "sleep-exit",
            {
                "model_name": trainer.model_name,
                "resume_at": next_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        self.ledger.patch(status="training")
        print("训练继续进行，任务状态已修改......")
        (
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.ema,
            cursor.start_epoch,
            cursor.batch_num,
        ) = load_training_state(
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.ema,
            trainer.saveDir,
        )
        print(f"Resuming training from epoch {cursor.start_epoch}")
        print("Now batch_num is: ", cursor.batch_num)
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


class TrainingLaunchWindow:
    def __init__(self, configfile):
        self.configfile = configfile

    def await_window(self):
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Current time: {current_time}")
            with open(self.configfile, "r", encoding="utf-8") as file:
                configs = json.load(file)
            if self._is_in_window(configs["allow_train_time_list"]):
                print("当前启动时间在允许训练的时间段内,立即开始训练......")
                return configs
            print("当前不在允许训练的时间段内, 进入等待期, 待至允许训练时段内自动开启训练任务...")
            next_start_time = calculate_next_training_time(
                configs["allow_train_time_list"]
            )
            if next_start_time:
                next_start_time = datetime.strptime(
                    next_start_time, "%Y-%m-%d %H:%M:%S"
                )
                time_until_next = (next_start_time - datetime.now()).total_seconds()
                print(
                    f"下一个允许训练的时间点为: {next_start_time.strftime('%Y-%m-%d %H:%M:%S')}. 当前进入等待期..."
                )
                time.sleep(max(time_until_next, 0))
            else:
                time.sleep(600)

    @staticmethod
    def _is_in_window(allow_train_time_list):
        current_time = datetime.now().strftime("%H:%M:%S")
        for start_time, end_time in allow_train_time_list:
            if start_time > end_time:
                if current_time >= start_time or current_time <= end_time:
                    return True
                continue
            if start_time <= current_time <= end_time:
                return True
        return False


warmup_policy_registry.register("polynomial-warmup", PolynomialWarmupRegulator)
checkpoint_policy_registry.register("evaluation-checkpoint", EvaluationCheckpointRelay)
time_window_policy_registry.register("temporal-sleep", TemporalSleepPolicy)
finalizer_policy_registry.register("encrypted-artifact-finalizer", EncryptedArtifactFinalizer)
