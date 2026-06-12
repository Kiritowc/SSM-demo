import os
from dataclasses import dataclass, field

from sunshink_cv.utils.tool import LoadYaml


@dataclass(frozen=True)
class RuntimePaths:
    yaml_path: str
    save_dir: str
    train_log_path: str
    names_path: str
    train_index_path: str
    val_index_path: str


@dataclass(frozen=True)
class ComponentDialect:
    loss: str = "detector-loss"
    evaluator: str = "coco-detection"
    dataset: str = "tensor-detection"
    optimizer: str = "sgd-momentum"
    scheduler: str = "multistep"
    exporter: str = "encrypted-onnx"
    warmup: str = "polynomial-warmup"
    checkpoint: str = "evaluation-checkpoint"
    time_window: str = "temporal-sleep"
    finalizer: str = "encrypted-artifact-finalizer"


@dataclass(frozen=True)
class TrainingRuntimeContext:
    opt: object
    cfg: object
    device: object
    paths: RuntimePaths
    dialect: ComponentDialect = field(default_factory=ComponentDialect)

    @classmethod
    def from_legacy_opt(cls, opt, yaml_path: str, device, save_dir: str):
        cfg = LoadYaml(yaml_path)
        normalized_save_dir = save_dir if save_dir.endswith("/") else save_dir + "/"
        os.makedirs(normalized_save_dir, exist_ok=True)
        paths = RuntimePaths(
            yaml_path=yaml_path,
            save_dir=normalized_save_dir,
            train_log_path=os.path.join(normalized_save_dir, "train_log.txt"),
            names_path=cfg.names,
            train_index_path=cfg.train_txt,
            val_index_path=cfg.val_txt,
        )
        return cls(opt=opt, cfg=cfg, device=device, paths=paths)
