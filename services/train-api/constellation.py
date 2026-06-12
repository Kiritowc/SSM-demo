import argparse
import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt

from cv.cfg import *
from cv.core.tasking import TaskArchiveRepository, TaskTopologyCompiler
from cv.export import convert_and_save_model
from cv.inference import ssDet
from cv.utils.tool import get_history_epoch


@dataclass(frozen=True)
class RuntimeCoordinate:
    model_name: str
    runs_dir: str

    @property
    def model_dir(self) -> str:
        return os.path.join(self.runs_dir, self.model_name)

    @property
    def best_weight_path(self) -> str:
        return os.path.join(self.model_dir, "best.bin")

    @property
    def runtime_weight_path(self) -> str:
        return os.path.join(self.model_dir, "run.bin")


class DatasetIngressFacade:
    def materialize(self, payload: Dict[str, str]) -> Dict[str, str]:
        required_fields = {"train", "test", "save_path"}
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        save_path = payload.get("save_path") or "./"
        os.makedirs(save_path, exist_ok=True)
        train_file_path = os.path.join(save_path, "train.txt")
        test_file_path = os.path.join(save_path, "test.txt")
        with open(train_file_path, "w") as file:
            file.write(payload["train"])
        with open(test_file_path, "w") as file:
            file.write(payload["test"])
        return {
            "message": "Files saved successfully",
            "train_path": train_file_path,
            "test_path": test_file_path,
        }


class TrainingTaskIngressFacade:
    def __init__(self):
        self.compiler = TaskTopologyCompiler()
        self.repository = TaskArchiveRepository()

    def enqueue(self, payload: Dict) -> Dict[str, str]:
        topology = self.compiler.compile(payload)
        task_file = self.repository.enqueue(topology)
        return {"message": "Training task added to queue", "task_file": task_file}


class TrainingObservatoryFacade:
    def read_task_status(self) -> str:
        with open(TASK_FILE, "r") as file:
            return file.read()

    def read_detailed_log(self) -> List[str]:
        with open(TRAIN_LOG, "r") as file:
            return [line.strip() for line in file.readlines() if line.strip()]

    def read_model_log(self, coordinate: RuntimeCoordinate) -> List[List[str]]:
        log_file = os.path.join(coordinate.model_dir, "train_log.txt")
        with open(log_file, "r") as file:
            return [line.strip().split(",") for line in file.readlines() if line.strip()]

    def read_history_epoch(self, coordinate: RuntimeCoordinate) -> int:
        return get_history_epoch(coordinate.model_dir + "/")

    def read_task_history(self) -> List[str]:
        with open(TASK_HISTORY, "r") as file:
            return file.readlines()

    def read_telemetry_stream(self, coordinate: RuntimeCoordinate) -> List[Dict]:
        telemetry_file = os.path.join(coordinate.model_dir, "telemetry.jsonl")
        return self._read_jsonl_stream(telemetry_file)

    def read_global_event_stream(self) -> List[Dict]:
        return self._read_jsonl_stream(EVENT_STREAM)

    def build_visual_log(self, coordinate: RuntimeCoordinate) -> bytes:
        log_file = os.path.join(coordinate.model_dir, "train_log.txt")
        with open(log_file, "r") as file:
            log_content = [line.strip() for line in file.readlines() if line.startswith("Epoch:")]

        epochs, ious, losses = [], [], []
        for line in log_content:
            epoch, iou, loss = line.strip().split(" ")
            epochs.append(int(epoch.split(":")[-1]))
            ious.append(float(iou.split(":")[-1]))
            losses.append(float(loss.split(":")[-1]))

        figure, (axis_iou, axis_loss) = plt.subplots(1, 2, figsize=(12, 6))
        axis_iou.plot(epochs, ious, label="IOU", color="blue")
        axis_iou.set_title("IOU over Epochs")
        axis_iou.set_xlabel("Epoch")
        axis_iou.set_ylabel("IOU")
        axis_iou.legend()
        axis_loss.plot(epochs, losses, label="Loss", color="red")
        axis_loss.set_title("Loss over Epochs")
        axis_loss.set_xlabel("Epoch")
        axis_loss.set_ylabel("Loss")
        axis_loss.legend()

        image_buffer = BytesIO()
        plt.savefig(image_buffer, format="png")
        plt.close(figure)
        image_buffer.seek(0)
        return image_buffer.getvalue()

    def stop_training(self) -> Dict[str, str]:
        with open(trainLogDir + "task_pid.txt", "r") as file:
            pid = int(file.read())
        os.kill(pid, 9)
        os.remove(trainLogDir + "task_pid.txt")
        with open(TRAIN_LOG, "a") as file:
            file.write(
                "\n"
                + "当前训练任务已终结! 训练进程号: "
                + str(pid)
                + ", 结束时间点: "
                + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                + "\n"
            )
        return {"message": "Training stopped"}

    def _read_jsonl_stream(self, target_path: str) -> List[Dict]:
        with open(target_path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file.readlines() if line.strip()]


class PredictionConstellationFacade:
    def __init__(self):
        self.temp_image_path = "temp_image.jpg"

    def predict(self, encoded_image: str, model_name: str, runs_dir: str) -> Dict:
        coordinate = RuntimeCoordinate(model_name=model_name, runs_dir=runs_dir or "runs/detect/")
        self._hydrate_image(encoded_image)
        self._ensure_runtime_artifact(coordinate)
        srcimg = cv2.imread(self.temp_image_path)
        model = ssDet(conf=0.5, nms=0.5, weight=coordinate.runtime_weight_path)
        res_data, rendered = model.detect(srcimg)
        cv2.imwrite("detection.jpg", rendered)
        _, encoded = cv2.imencode(".jpg", rendered)
        return {
            "resData": res_data,
            "resImage": base64.b64encode(encoded.tobytes()).decode("utf-8"),
        }

    def _hydrate_image(self, encoded_image: str) -> None:
        image_data = base64.b64decode(encoded_image)
        with open(self.temp_image_path, "wb") as file:
            file.write(image_data)

    def _ensure_runtime_artifact(self, coordinate: RuntimeCoordinate) -> None:
        if not os.path.exists(coordinate.best_weight_path):
            raise FileNotFoundError("No model file founded")
        if os.path.exists(coordinate.runtime_weight_path):
            return
        opt = argparse.Namespace(
            model=coordinate.model_name,
            weight=coordinate.best_weight_path,
            save_path=coordinate.runtime_weight_path,
            img=self.temp_image_path,
        )
        convert_and_save_model(opt)
