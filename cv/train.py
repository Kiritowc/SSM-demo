import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)
import sysconfig
from datetime import datetime

def _prefer_env_site_packages():
    """Prefer packages installed in the active env over ~/.local user packages."""
    env_site_packages = sysconfig.get_paths().get("purelib")
    if not env_site_packages or env_site_packages not in sys.path:
        return

    sys.path.remove(env_site_packages)
    insert_at = 1 if sys.path else 0
    sys.path.insert(insert_at, env_site_packages)


_prefer_env_site_packages()

from cv.cfg import taskCfgDir
from cv.core.tasking import TaskConfigMaterializer, TaskTopologyCompiler
from cv.core.training import TrainingLaunchWindow


class LegacyTrainBootstrap:
    def __init__(self, configfile):
        self.configfile = configfile

    @staticmethod
    def _next_archive_dir(runs_dir, model_name):
        base_name = f"{model_name}_{datetime.now().strftime('%Y-%m-%d')}"
        candidate = os.path.join(runs_dir, base_name)
        index = 1
        while os.path.exists(candidate):
            candidate = os.path.join(runs_dir, f"{base_name}_{index}")
            index += 1
        return candidate

    def launch(self):
        configs = TrainingLaunchWindow(self.configfile).await_window()
        self._materialize_task_config(configs)

        from cv.ssDet import sunshinkDet

        model_name = configs["model_name"]
        epochs = configs["cfg_yaml"]["TRAIN"]["END_EPOCH"]
        runs_dir = configs["runsDir"]
        if configs.get("zerostart"):
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
                    self._next_archive_dir(archive_root, tag),
                )
        model = sunshinkDet(
            yaml=os.path.join("cv", "configs", "self.yaml"),
            model=model_name,
            weight=configs.get("pretrained_weight"),
            epochs=epochs,
            dir=runs_dir,
            save_dir=configs.get("save_dir"),
        )
        model.train()
        print("训练任务完成!")
        gc.collect()
        return {"model_name": model_name, "runs_dir": runs_dir}

    @staticmethod
    def _materialize_task_config(configs):
        topology = TaskTopologyCompiler().compile(configs)
        TaskConfigMaterializer().materialize(topology)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configfile",
        type=str,
        default=os.path.join(taskCfgDir, "task_config.json"),
        help="config file",
    )
    opt = parser.parse_args()
    LegacyTrainBootstrap(opt.configfile).launch()
