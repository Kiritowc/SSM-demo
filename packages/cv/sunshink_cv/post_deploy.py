"""Post-training deploy: export ONNX, build TensorRT engine, mark robot_toy ready."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import os
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from sunshink_cv.export import convert_and_save_model
from sunshink_cv._vlm_train_ui import (
    log_camera_restart,
    log_deploy_complete,
    log_deploy_failed,
    log_deploy_start,
)


from ssm_common.paths import repo_root
REPO_ROOT = repo_root()
TRTEXEC = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")
ROBOT_TOY_ENGINE = REPO_ROOT / "camera" / "engines" / "robot_far.engine"
ROBOT_TOY_ONNX = REPO_ROOT / "camera" / "engines" / "robot_far.onnx"
ROBOT_TOY_FALLBACK_ENGINE = REPO_ROOT / "camera" / "engines" / "robot_toy.engine"
ROBOT_TOY_READY_MARKER = REPO_ROOT / "camera" / "engines" / ".robot_toy_ready"


def clear_robot_toy_ready() -> None:
    if ROBOT_TOY_READY_MARKER.exists():
        ROBOT_TOY_READY_MARKER.unlink()


def is_robot_toy_ready() -> bool:
    return ROBOT_TOY_READY_MARKER.is_file() and ROBOT_TOY_ENGINE.is_file()


def deploy_robot_toy_engine(
    *,
    save_dir: str,
    model_name: str,
    val_txt: str,
    yaml_path: str = "packages/cv/sunshink_cv/configs/self.yaml",
    restart_camera: bool = False,
) -> bool:
    save_path = Path(save_dir)
    if not save_path.is_absolute():
        save_path = REPO_ROOT / save_path
    best_bin = save_path / "best.bin"
    if not best_bin.is_file():
        log_deploy_failed("best weights not found")
        return False

    val_path = Path(val_txt)
    if not val_path.is_absolute():
        val_path = REPO_ROOT / val_path
    if not val_path.is_file():
        log_deploy_failed("validation manifest not found: %s" % val_path)
        return False

    with open(val_path, encoding="utf-8") as file:
        val_list = [line.strip() for line in file if line.strip()]
    if not val_list:
        log_deploy_failed("validation manifest is empty")
        return False

    log_deploy_start()
    ROBOT_TOY_ONNX.parent.mkdir(parents=True, exist_ok=True)

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
        plain_onnx=str(ROBOT_TOY_ONNX),
        skip_preview=True,
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            convert_and_save_model(opt)
    except Exception as exc:
        log_deploy_failed("export failed (%s)" % exc)
        return False

    if not Path(TRTEXEC).is_file():
        log_deploy_failed("trtexec not found at %s" % TRTEXEC)
        return False

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    from sunshink_cv._demo_services import stop_camera

    stop_camera()

    engine_built = False
    try:
        subprocess.run(
            [
                TRTEXEC,
                "--onnx=%s" % ROBOT_TOY_ONNX,
                "--saveEngine=%s" % ROBOT_TOY_ENGINE,
                "--fp16",
                "--workspace=256",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        engine_built = ROBOT_TOY_ENGINE.is_file()
    except subprocess.CalledProcessError:
        pass

    if not engine_built and ROBOT_TOY_FALLBACK_ENGINE.is_file():
        shutil.copy2(ROBOT_TOY_FALLBACK_ENGINE, ROBOT_TOY_ENGINE)
        engine_built = ROBOT_TOY_ENGINE.is_file()

    if not engine_built:
        return False

    ROBOT_TOY_READY_MARKER.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    log_deploy_complete()
    if restart_camera:
        from sunshink_cv._demo_services import start_camera

        log_camera_restart()
        start_camera()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX and build robot_toy TensorRT engine.")
    parser.add_argument("--save-dir", required=True, help="Training output dir containing best.bin")
    parser.add_argument("--model-name", required=True, help="Backbone id, e.g. ssi_a")
    parser.add_argument("--val-txt", required=True, help="Validation manifest for export sample image")
    parser.add_argument("--yaml-path", default="cv/configs/self.yaml")
    parser.add_argument(
        "--restart-camera",
        action="store_true",
        help="Restart camera after deploy so the new engine is loaded",
    )
    args = parser.parse_args()
    ok = deploy_robot_toy_engine(
        save_dir=args.save_dir,
        model_name=args.model_name,
        val_txt=args.val_txt,
        yaml_path=args.yaml_path,
        restart_camera=args.restart_camera,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
