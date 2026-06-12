import io
from copy import copy
from dataclasses import dataclass

import torch
from torch import optim

from sunshink_cv.core.detector import Detector
from sunshink_cv.core.loss import DetectorLoss
from sunshink_cv.core.registry import NamedComponentRegistry
from sunshink_cv.utils.datasets import TensorDataset, collate_fn
from sunshink_cv.utils.evaluation import CocoDetectionEvaluator
from sunshink_cv.utils.tool import EMA
from sunshink_cv.core.backbone import MoEn 


optimizer_registry = NamedComponentRegistry("optimizer")
scheduler_registry = NamedComponentRegistry("scheduler")
loss_registry = NamedComponentRegistry("loss")
evaluator_registry = NamedComponentRegistry("evaluator")
dataset_registry = NamedComponentRegistry("dataset")


def _build_sgd_momentum(runtime, model):
    return optim.SGD(
        params=model.parameters(),
        lr=runtime.cfg.learn_rate,
        momentum=0.949,
        weight_decay=0.0005,
    )


def _build_multistep(runtime, optimizer):
    return optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=runtime.cfg.milestones, gamma=0.1
    )


def _build_detector_loss(runtime):
    return DetectorLoss(runtime.device)


def _build_coco_detection(runtime):
    return CocoDetectionEvaluator(runtime.paths.names_path, runtime.device)


def _build_tensor_dataset(runtime, split):
    if split == "train":
        return TensorDataset(
            runtime.paths.train_index_path,
            runtime.cfg.input_width,
            runtime.cfg.input_height,
            runtime.opt,
        )
    val_opt = copy(runtime.opt)
    val_opt.aug = False
    return TensorDataset(
        runtime.paths.val_index_path,
        runtime.cfg.input_width,
        runtime.cfg.input_height,
        val_opt,
    )


optimizer_registry.register("sgd-momentum", _build_sgd_momentum)
scheduler_registry.register("multistep", _build_multistep)
loss_registry.register("detector-loss", _build_detector_loss)
evaluator_registry.register("coco-detection", _build_coco_detection)
dataset_registry.register("tensor-detection", _build_tensor_dataset)


@dataclass
class TrainingComponentBundle:
    model: object
    optimizer: object
    scheduler: object
    ema: object
    loss_function: object
    evaluation: object
    train_dataset: object
    val_dataset: object
    train_dataloader: object
    val_dataloader: object


class DetectorPlatformAssembly:
    def __init__(self):
        self.vault = MoEn()

    def build(self, runtime):
        num_workers = getattr(runtime.opt, "num_workers", 1)
        model = Detector(runtime.cfg.category_num, runtime.opt, runtime.opt.weight is not None).to(
            runtime.device
        )
        if runtime.opt.weight is not None:
            print("load weight from:%s" % runtime.opt.weight)
            payload = self._load_weight_state(runtime.opt.weight, runtime.device)
            try:
                model.load_state_dict(payload)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "size mismatch" in msg or "loading state_dict" in msg:
                    self._graft_detector_weights(model, payload)
                else:
                    raise

        optimizer = optimizer_registry.build(runtime.dialect.optimizer, runtime, model)
        scheduler = scheduler_registry.build(runtime.dialect.scheduler, runtime, optimizer)
        ema = EMA(model, decay=0.9999)
        loss_function = loss_registry.build(runtime.dialect.loss, runtime)
        evaluation = evaluator_registry.build(runtime.dialect.evaluator, runtime)
        train_dataset = dataset_registry.build(runtime.dialect.dataset, runtime, "train")
        val_dataset = dataset_registry.build(runtime.dialect.dataset, runtime, "val")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=runtime.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=runtime.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
        return TrainingComponentBundle(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            loss_function=loss_function,
            evaluation=evaluation,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

    def _load_weight_state(self, weight_path, device):
        try:
            decrypted_data = self.vault.de_model_to_memory(weight_path)
            buffer = io.BytesIO(decrypted_data)
            return torch.load(buffer, map_location=device)
        except Exception:
            return torch.load(weight_path, map_location=device)

    @staticmethod
    def _unwrap_model_state(checkpoint_payload):
        if not isinstance(checkpoint_payload, dict):
            return checkpoint_payload
        if "model_state_dict" in checkpoint_payload:
            return checkpoint_payload["model_state_dict"]
        if "state_dict" in checkpoint_payload:
            return checkpoint_payload["state_dict"]
        return checkpoint_payload

    def _graft_detector_weights(self, model, checkpoint_payload):
        state_dict = self._unwrap_model_state(checkpoint_payload)
        model_dict = model.state_dict()
        grafted = {}
        for key, tensor in state_dict.items():
            if key not in model_dict:
                continue
            target = model_dict[key]
            if target.shape != tensor.shape:
                continue
            grafted[key] = tensor.to(device=target.device, dtype=target.dtype)
        model_dict.update(grafted)
        model.load_state_dict(model_dict)
