"""Post-training deploy: ONNX export + TensorRT engine build."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import os
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from ssdet.core.exporting import convert_and_save_model
from ssdet.paths import RUNTIME_YAML, deploy_config
from ssm.config import platform_binaries
from ssm.paths import repo_root

REPO_ROOT = repo_root()


def _trtexec() -> str:
    return os.environ.get(
        "TRTEXEC",
        platform_binaries().get("trtexec", "/usr/src/tensorrt/bin/trtexec"),
    )


def _deploy_paths() -> dict[str, Path]:
    deploy = deploy_config()
    engine = REPO_ROOT / deploy["engine"]
    return {
        "engine": engine,
        "onnx": REPO_ROOT / deploy["onnx"],
        "fallback_engine": engine,
        "marker": REPO_ROOT / deploy["marker"],
    }


def _say(msg: str) -> None:
    print("[ssdet] %s" % msg, flush=True)


def clear_robot_toy_ready() -> None:
    marker = _deploy_paths()["marker"]
    if marker.exists():
        marker.unlink()


def is_robot_toy_ready() -> bool:
    paths = _deploy_paths()
    return paths["marker"].is_file() and paths["engine"].is_file()


def deploy_robot_toy_engine(
    *,
    save_dir: str,
    model_name: str,
    val_txt: str,
    yaml_path: str = RUNTIME_YAML,
) -> bool:
    paths = _deploy_paths()
    save_path = Path(save_dir)
    if not save_path.is_absolute():
        save_path = REPO_ROOT / save_path
    best_bin = save_path / "best.bin"
    if not best_bin.is_file():
        return False

    val_path = Path(val_txt)
    if not val_path.is_absolute():
        val_path = REPO_ROOT / val_path
    if not val_path.is_file():
        return False

    with open(val_path, encoding="utf-8") as file:
        val_list = [line.strip() for line in file if line.strip()]
    if not val_list:
        return False

    _say("正在同步模型...")
    paths["onnx"].parent.mkdir(parents=True, exist_ok=True)

    opt = argparse.Namespace(
        yaml=str(REPO_ROOT / yaml_path) if not Path(yaml_path).is_absolute() else yaml_path,
        model=model_name,
        weight=str(best_bin),
        save_path=str(save_path / "run.bin"),
        img=random.choice(val_list),
        thresh=0.50,
        spp="spp",
        ins=None,
        ous=None,
        plain_onnx=str(paths["onnx"]),
        skip_preview=True,
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            convert_and_save_model(opt)
    except Exception:
        return False

    trtexec = _trtexec()
    if not Path(trtexec).is_file():
        return False

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    engine_built = False
    try:
        subprocess.run(
            [
                trtexec,
                "--onnx=%s" % paths["onnx"],
                "--saveEngine=%s" % paths["engine"],
                "--fp16",
                "--workspace=256",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        engine_built = paths["engine"].is_file()
    except subprocess.CalledProcessError:
        pass

    if not engine_built and paths["fallback_engine"].is_file():
        shutil.copy2(paths["fallback_engine"], paths["engine"])
        engine_built = paths["engine"].is_file()

    if not engine_built:
        return False

    paths["marker"].write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    _say("模型已就绪。")
    return True


def deploy_from_config() -> bool:
    from ssdet.cfg import task_config

    return deploy_robot_toy_engine(
        save_dir=task_config["save_dir"],
        model_name=task_config["model_name"],
        val_txt=task_config["cfg_yaml"]["DATASET"]["VAL"],
    )
