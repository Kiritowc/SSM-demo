"""Training engine with checkpointing and early stopping."""



from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ets.tasks.factory import build_task
from ets.utils.checkpoint import (
    best_checkpoint_path,
    last_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from ets.utils.distributed import is_main_process
from ets.utils.logger import (
    DATE_FMT,
    TRAIN_FMT,
    TrainFormatter,
    format_epoch_line,
    log_eval_metrics,
    log_plain,
    resolve_logger,
)
from ets.utils.visualizer import visualize_training


class EarlyStopping:

    """Early stopping on monitored metric."""



    def __init__(self, patience: int = 10, mode: str = "min") -> None:

        self.patience = patience

        self.mode = mode

        self.counter = 0

        self.best_score: float | None = None

        self.should_stop = False



    def step(self, score: float) -> bool:

        if self.best_score is None:

            self.best_score = score

            return False



        improved = score < self.best_score if self.mode == "min" else score > self.best_score

        if improved:

            self.best_score = score

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.should_stop = True

        return self.should_stop



    def state_dict(self) -> dict[str, Any]:

        return {

            "patience": self.patience,

            "mode": self.mode,

            "counter": self.counter,

            "best_score": self.best_score,

            "should_stop": self.should_stop,

        }



    def load_state_dict(self, state: dict[str, Any]) -> None:

        self.patience = int(state.get("patience", self.patience))

        self.mode = str(state.get("mode", self.mode))

        self.counter = int(state.get("counter", 0))

        self.best_score = state.get("best_score")

        self.should_stop = bool(state.get("should_stop", False))





class Trainer:

    """Main training loop."""



    def __init__(

        self,

        model: torch.nn.Module,

        cfg: dict[str, Any],

        train_loader,

        val_loader,

        device: torch.device,

        run_dir: Path,

        scaler=None,

        target_scaler=None,

        logger: logging.Logger | None = None,

    ) -> None:

        self.model = model

        self.cfg = cfg

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.device = device

        self.run_dir = Path(run_dir)

        self.feature_scaler = scaler

        self.target_scaler = target_scaler

        self.task = build_task(cfg, target_scaler=target_scaler)

        self.logger = resolve_logger(logger, name="ets", style="train")
        log_path = self.run_dir / "train.log"
        has_file = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == log_path.resolve()
            for handler in self.logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(TrainFormatter(fmt=TRAIN_FMT, datefmt=DATE_FMT))
            self.logger.addHandler(file_handler)



        train_cfg = cfg["train"]

        self.epochs = int(train_cfg["epochs"])

        self.log_interval = int(train_cfg.get("log_interval", 10))

        lr = float(train_cfg["lr"])

        weight_decay = float(train_cfg.get("weight_decay", 0.0))

        optimizer_name = train_cfg.get("optimizer", "adam").lower()



        if optimizer_name == "adamw":

            self.optimizer = torch.optim.AdamW(

                model.parameters(), lr=lr, weight_decay=weight_decay

            )

        else:

            self.optimizer = torch.optim.Adam(

                model.parameters(), lr=lr, weight_decay=weight_decay

            )



        self.scheduler = self._build_scheduler(train_cfg)



        es_cfg = train_cfg.get("early_stopping", {})

        self.early_stopping = None

        if es_cfg.get("enabled", True):

            self.early_stopping = EarlyStopping(

                patience=int(es_cfg.get("patience", 10)),

                mode=self.task.monitor_mode,

            )



        self.monitor_name = self.task.monitor_name

        self.best_monitor_score = float("inf") if self.task.monitor_mode == "min" else float("-inf")

        self.start_epoch = 1



        resume_path = train_cfg.get("resume")

        if resume_path:

            self._load_resume(Path(resume_path))



        self.logger.info("监控指标: %s (mode=%s)", self.monitor_name, self.task.monitor_mode)



    def _load_resume(self, resume_path: Path) -> None:

        if not resume_path.exists():

            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")



        state = load_checkpoint(

            resume_path,

            self.model,

            self.optimizer,

            self.scheduler,

            device=self.device,

        )

        self.start_epoch = int(state.get("epoch", 0)) + 1

        self.best_monitor_score = float(

            state.get("best_monitor_score", self.best_monitor_score)

        )



        if self.early_stopping and "early_stopping" in state:

            self.early_stopping.load_state_dict(state["early_stopping"])



        self.logger.info(

            "从 checkpoint 恢复训练: %s (从第 %d 轮继续)",

            resume_path,

            self.start_epoch,

        )



    def _display_metrics(self, metrics: dict[str, float]) -> dict[str, float]:

        """Use original-scale loss for eval logs when RMSE is available."""

        display = dict(metrics)

        if self.cfg["task"]["type"] == "forecast" and "rmse" in display:

            display["loss"] = float(display["rmse"] ** 2)

        return display



    def _build_scheduler(self, train_cfg: dict[str, Any]):

        scheduler_name = train_cfg.get("scheduler")

        if scheduler_name is None:

            return None

        params = train_cfg.get("scheduler_params", {})

        name = scheduler_name.lower()

        if name == "cosine":

            t_max = int(params.get("T_max", train_cfg["epochs"]))

            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)

        if name == "step":

            step_size = int(params.get("step_size", 10))

            gamma = float(params.get("gamma", 0.5))

            return torch.optim.lr_scheduler.StepLR(

                self.optimizer, step_size=step_size, gamma=gamma

            )

        return None



    def _run_epoch(self, train: bool = True) -> dict[str, float]:

        loader = self.train_loader if train else self.val_loader

        self.model.train(train)



        total_loss = 0.0

        metric_sums: dict[str, float] = {}

        num_batches = 0



        for batch in loader:

            if train:

                self.optimizer.zero_grad()

                step_fn = self.task.training_step

            else:

                step_fn = self.task.validation_step



            result = step_fn(self.model, batch, self.device)

            loss = result["loss"]



            if train:

                loss.backward()

                self.optimizer.step()



            total_loss += loss.item()

            for key, value in result["metrics"].items():

                metric_sums[key] = metric_sums.get(key, 0.0) + value

            num_batches += 1



        avg_loss = total_loss / max(num_batches, 1)

        avg_metrics = {k: v / max(num_batches, 1) for k, v in metric_sums.items()}

        avg_metrics["loss"] = avg_loss

        return avg_metrics



    def _checkpoint_extra(self) -> dict[str, Any]:

        extra: dict[str, Any] = {"best_monitor_score": self.best_monitor_score}

        if self.early_stopping is not None:

            extra["early_stopping"] = self.early_stopping.state_dict()

        return extra



    def fit(self) -> dict[str, Any]:

        """Run full training loop."""

        self.logger.info("开始训练，共 %d 轮", self.epochs)

        history: list[dict[str, Any]] = []



        for epoch in range(self.start_epoch, self.epochs + 1):

            train_metrics = self._run_epoch(train=True)

            val_metrics = self._run_epoch(train=False)



            if self.scheduler is not None:

                self.scheduler.step()



            val_loss = val_metrics["loss"]

            train_loss = train_metrics["loss"]

            _, monitor_score = self.task.resolve_monitor_score(val_metrics)



            record = {

                "epoch": epoch,

                "train": train_metrics,

                "val": val_metrics,

                "lr": self.optimizer.param_groups[0]["lr"],

                "monitor": self.monitor_name,

                "monitor_score": monitor_score,

            }

            history.append(record)



            is_best = self.task.is_better(monitor_score, self.best_monitor_score)

            log_plain(

                self.logger,

                format_epoch_line(epoch, self.epochs, train_loss, val_loss),

            )



            is_last_epoch = epoch == self.epochs

            should_log_eval = epoch % self.log_interval == 0 or is_last_epoch

            if should_log_eval and val_metrics.keys() - {"loss"}:

                log_eval_metrics(self.logger, self._display_metrics(val_metrics))



            save_checkpoint(

                last_checkpoint_path(self.run_dir),

                self.model,

                self.optimizer,

                epoch,

                val_metrics,

                self.cfg,

                scaler=self.feature_scaler,

                target_scaler=self.target_scaler,

                scheduler=self.scheduler,

                extra=self._checkpoint_extra(),

            )



            if is_best:

                self.best_monitor_score = monitor_score

                save_checkpoint(

                    best_checkpoint_path(self.run_dir),

                    self.model,

                    self.optimizer,

                    epoch,

                    val_metrics,

                    self.cfg,

                    scaler=self.feature_scaler,

                    target_scaler=self.target_scaler,

                    scheduler=self.scheduler,

                    extra=self._checkpoint_extra(),

                )



            if self.early_stopping and self.early_stopping.step(monitor_score):

                if not should_log_eval and val_metrics.keys() - {"loss"}:

                    log_eval_metrics(self.logger, self._display_metrics(val_metrics))

                log_plain(self.logger, f"早停于第 {epoch} 轮 (监控: {self.monitor_name})")

                break



        if is_main_process():

            visualize_training(history, self.run_dir, self.cfg, logger=self.logger)



        return {

            "best_monitor_score": self.best_monitor_score,

            "monitor": self.monitor_name,

            "history": history,

        }


