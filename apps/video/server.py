#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import mimetypes
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool 

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo

REPO_ROOT = str(bootstrap_repo(_REPO))
from ssm.paths import get_paths

UI_ROOT = str(get_paths().apps_web)

from cv.paths import ROBOT_TOY_CLASSES, ROBOT_TOY_ENGINE
from ssm.config import platform_services
from vlm.prompts import (
    _assistant_content_message,
    _delta_content_piece,
    _image_data_url_from_jpeg_bytes,
    build_vlm_rules,
    build_vlm_system_prompt,
    should_stop_vlm_repeat,
)


class LatestFrame:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._seq = 0
        self._frame = None
        self._running = True

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def update(self, frame) -> None:
        with self._cond:
            self._seq += 1
            self._frame = frame
            self._cond.notify_all()

    def wait_next(self, seen_seq: int, timeout: Optional[float]) -> Tuple[int, Any]:
        with self._cond:
            while self._running and self._seq == seen_seq:
                if not self._cond.wait(timeout):
                    return seen_seq, None
            if not self._running:
                return seen_seq, None
            return self._seq, self._frame


class LatestJPEG:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._seq = 0
        self._jpeg = None
        self._stats: Dict[str, Any] = {}
        self._cv_payload: Dict[str, Any] = {}
        self._running = True

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def update(
        self,
        jpeg: bytes,
        stats: Dict[str, Any],
        cv_payload: Dict[str, Any],
    ) -> None:
        with self._cond:
            self._seq += 1
            self._jpeg = jpeg
            self._stats = dict(stats)
            self._cv_payload = dict(cv_payload)
            self._cond.notify_all()

    def stats(self) -> Dict[str, Any]:
        with self._cond:
            return dict(self._stats)

    def cv_payload(self) -> Dict[str, Any]:
        with self._cond:
            return dict(self._cv_payload)

    def latest_jpeg(self) -> Optional[bytes]:
        with self._cond:
            return self._jpeg

    def wait_next(self, seen_seq: int, timeout: Optional[float]) -> Tuple[int, Optional[bytes]]:
        with self._cond:
            while self._running and self._seq == seen_seq:
                if not self._cond.wait(timeout):
                    return seen_seq, None
            if not self._running:
                return seen_seq, None
            return self._seq, self._jpeg


class CameraReader(threading.Thread):
    def __init__(self, args: argparse.Namespace, frames: LatestFrame) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.frames = frames
        self.running = True

    def run(self) -> None:
        cap = cv2.VideoCapture(self.args.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera: %s" % self.args.device)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                self.frames.update(frame)
        finally:
            cap.release()
            self.frames.stop()

    def stop(self) -> None:
        self.running = False


def make_grid(nx: int = 26, ny: int = 26) -> np.ndarray:
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def draw_pred(frame, class_name: str, conf: float, left: int, top: int, right: int, bottom: int):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
    label = "%s:%.2f" % (class_name, conf)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
    return frame


def resdata_to_objects(res_data: dict):
    objects = []
    det_id = 0
    for label in sorted(res_data.keys()):
        for item in res_data[label]:
            one_label, one_box, _one_score = item
            bbox_xyxy = [
                int(one_box[0]),
                int(one_box[1]),
                int(one_box[2]),
                int(one_box[3]),
            ]
            objects.append(
                {
                    "id": det_id,
                    "label": str(one_label),
                    "bbox_xyxy": bbox_xyxy,
                }
            )
            det_id += 1
    return objects


def build_cv_payload(frame, res_data: dict, device: str):
    h, w = frame.shape[:2]
    return {
        "objects": resdata_to_objects(res_data),
        "image_width": int(w),
        "image_height": int(h),
    }


class CvModelState:
    def __init__(self, args: argparse.Namespace) -> None:
        self._lock = threading.Lock()
        self._current = args.default_cv_model
        self._specs = {
            "none": {"label": "默认", "engine": "", "names": ""},
            "ssg_a_robot_toy": {
                "label": "robot_toy",
                "engine": args.robot_toy_engine,
                "names": args.robot_toy_names,
            },
        }
        if self._current not in self._specs:
            self._current = "none"

    def get(self) -> str:
        with self._lock:
            return self._current

    def set(self, name: str) -> None:
        if name not in self._specs:
            raise ValueError("Unknown CV model: %s" % name)
        with self._lock:
            self._current = name

    def spec(self, name: str) -> Dict[str, str]:
        return dict(self._specs[name])

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            current = self._current
        return {
            "current": current,
            "models": [
                {"id": key, "label": spec["label"]}
                for key, spec in self._specs.items()
            ],
        }


def postprocess(frame, outs: np.ndarray, classes, conf_threshold: float, nms_threshold: float):
    frame_height, frame_width = frame.shape[:2]
    class_ids = []
    confidences = []
    boxes = []
    res_data = {}

    outs = outs.squeeze(axis=0).reshape(len(classes) + 5, -1).T
    outs[:, 3:5] = sigmoid(outs[:, 3:5])
    grid = make_grid(26, 26)
    outs[:, 1:3] = (np.tanh(outs[:, 1:3]) + grid) / np.tile(np.array([26, 26]), (outs.shape[0], 1))

    for detection in outs:
        scores = detection[5:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id] * detection[0])
        if confidence > conf_threshold:
            center_x = int(detection[1] * frame_width)
            center_y = int(detection[2] * frame_height)
            width = int(detection[3] * frame_width)
            height = int(detection[4] * frame_height)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            class_ids.append(class_id)
            confidences.append(confidence)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) == 0:
        return res_data, frame
    for i in np.array(indices).reshape(-1):
        box = boxes[int(i)]
        left = max(int(box[0]), 0)
        top = max(int(box[1]), 0)
        width = int(box[2])
        height = int(box[3])
        label = classes[class_ids[int(i)]]
        score = float(confidences[int(i)])
        one_box = [left, top, left + width, top + height]
        frame = draw_pred(frame, label, score, left, top, left + width, top + height)
        res_data.setdefault(label, []).append([label, one_box, score])
    return res_data, frame


class TensorRTRunner:
    backend = "tensorrt-fp16"

    def __init__(self, args: argparse.Namespace) -> None:
        system_dist = "/usr/lib/python3.8/dist-packages"
        if system_dist not in sys.path:
            sys.path.append(system_dist)
        import tensorrt as trt  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel

        self.args = args
        self.classes = [
            line.strip()
            for line in open(os.path.join(REPO_ROOT, args.names), "r", encoding="utf-8").readlines()
            if line.strip()
        ]
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(args.engine, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine: %s" % args.engine)
        self.context = self.engine.create_execution_context()
        self.cudart = ctypes.CDLL("libcudart.so")
        self._init_cuda_api()
        self.bindings = [0] * self.engine.num_bindings
        self.host_outputs = {}
        self.device_allocs = []
        self.input_index = None
        self.output_index = None
        self._allocate_bindings()

    def _init_cuda_api(self) -> None:
        self.cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cudart.cudaMalloc.restype = ctypes.c_int
        self.cudart.cudaFree.argtypes = [ctypes.c_void_p]
        self.cudart.cudaFree.restype = ctypes.c_int
        self.cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.cudart.cudaMemcpy.restype = ctypes.c_int

    def _cuda_check(self, code: int, label: str) -> None:
        if code != 0:
            raise RuntimeError("%s failed with CUDA error %d" % (label, code))

    def _malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._cuda_check(self.cudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        self.device_allocs.append(ptr)
        return ptr

    def _allocate_bindings(self) -> None:
        trt = self.trt
        for idx in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            device_ptr = self._malloc(nbytes)
            self.bindings[idx] = int(device_ptr.value)
            if self.engine.binding_is_input(idx):
                self.input_index = idx
            else:
                self.output_index = idx
                self.host_outputs[idx] = np.empty(shape, dtype=dtype)
        if self.input_index is None or self.output_index is None:
            raise RuntimeError("Unexpected TensorRT engine bindings")

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.args.input_size, self.args.input_size))
        blob = np.ascontiguousarray(blob.astype(np.float32))
        input_ptr = ctypes.c_void_p(self.bindings[self.input_index])
        output_ptr = ctypes.c_void_p(self.bindings[self.output_index])
        output = self.host_outputs[self.output_index]
        self._cuda_check(
            self.cudart.cudaMemcpy(input_ptr, ctypes.c_void_p(blob.ctypes.data), blob.nbytes, 1),
            "cudaMemcpy H2D",
        )
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("TensorRT execute_v2 failed")
        self._cuda_check(
            self.cudart.cudaMemcpy(ctypes.c_void_p(output.ctypes.data), output_ptr, output.nbytes, 2),
            "cudaMemcpy D2H",
        )
        return postprocess(frame, output, self.classes, self.args.conf, self.args.nms)

    def close(self) -> None:
        for ptr in self.device_allocs:
            self.cudart.cudaFree(ptr)
        self.device_allocs = []


def create_runner(args: argparse.Namespace, spec: Dict[str, str]):
    runner_args = argparse.Namespace(**vars(args))
    runner_args.engine = spec["engine"]
    runner_args.names = spec["names"]
    return TensorRTRunner(runner_args)


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        args: argparse.Namespace,
        frames: LatestFrame,
        output: LatestJPEG,
        cv_state: CvModelState,
    ) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.frames = frames
        self.output = output
        self.cv_state = cv_state
        self.running = True

    def run(self) -> None:
        os.chdir(REPO_ROOT)
        model = None
        loaded_cv = None

        seen_seq = 0
        processed = 0
        t0 = time.monotonic()
        last_emit = 0.0
        min_emit_interval = 1.0 / self.args.output_fps if self.args.output_fps > 0 else 0.0

        while self.running:
            seen_seq, frame = self.frames.wait_next(seen_seq, timeout=1.0)
            if frame is None:
                continue

            infer_start = time.monotonic()
            current_cv = self.cv_state.get()
            if current_cv == "none":
                res_data = {}
                annotated = frame.copy()
                infer_ms = 0.0
                if model is not None:
                    model.close()
                    model = None
                    loaded_cv = None
            else:
                if model is None or loaded_cv != current_cv:
                    if model is not None:
                        model.close()
                    spec = self.cv_state.spec(current_cv)
                    model = create_runner(self.args, spec)
                    loaded_cv = current_cv
                frame_annotated = frame.copy()
                res_data, annotated = model.detect(frame_annotated)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
            processed += 1
            elapsed = max(time.monotonic() - t0, 1e-6)
            infer_fps = processed / elapsed

            now = time.monotonic()
            if min_emit_interval and now - last_emit < min_emit_interval:
                continue
            last_emit = now

            stats = {
                "backend": model.backend if model is not None else "none",
                "cv_model": current_cv,
                "detections": sum(len(v) for v in res_data.values()),
                "infer_ms": round(infer_ms, 1),
                "infer_fps": round(infer_fps, 2),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            }
            cv_payload = build_cv_payload(annotated, res_data, self.args.device)
            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality]
            )
            if ok:
                self.output.update(encoded.tobytes(), stats, cv_payload)

        if model is not None:
            model.close()
        self.output.stop()

    def stop(self) -> None:
        self.running = False


class MJPEGServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16


class AskState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 0
        self._response = None

    def begin(self):
        with self._lock:
            self._next_id += 1
            my_id = self._next_id
            prev = self._response
            self._response = None
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass
        return my_id

    def register_response(self, my_id, resp) -> bool:
        with self._lock:
            if self._next_id != my_id:
                close = True
            else:
                self._response = resp
                close = False
        if close:
            try:
                resp.close()
            except Exception:
                pass
            return False
        return True

    def clear(self, my_id) -> None:
        with self._lock:
            if self._next_id == my_id:
                self._response = None

    def is_current(self, my_id) -> bool:
        with self._lock:
            return self._next_id == my_id


ASK_STATE = AskState()


def post_chat_completions_cancelable(
    base_url: str,
    body: Dict[str, Any],
    timeout: float,
    state: "AskState",
    my_id: int,
) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP %d: %s" % (e.code, err_body)) from e
    except urllib.error.URLError as e:
        raise RuntimeError("Request failed (is llama-server running?): %s" % e) from e

    if not state.register_response(my_id, resp):
        raise RuntimeError("CANCELLED")

    try:
        try:
            payload = resp.read()
        except Exception as exc:
            if not state.is_current(my_id):
                raise RuntimeError("CANCELLED") from exc
            raise
    finally:
        state.clear(my_id)

    return json.loads(payload.decode("utf-8"))


def post_chat_completions_stream_cancelable(
    base_url: str,
    body: Dict[str, Any],
    timeout: float,
    state: "AskState",
    my_id: int,
):
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = dict(body)
    payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP %d: %s" % (e.code, err_body)) from e
    except urllib.error.URLError as e:
        raise RuntimeError("Request failed (is llama-server running?): %s" % e) from e

    if not state.register_response(my_id, resp):
        raise RuntimeError("CANCELLED")

    try:
        while state.is_current(my_id):
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            err = obj.get("error")
            if err is not None:
                raise RuntimeError("Stream error: %s" % err)
            piece = _delta_content_piece(obj)
            if piece:
                yield piece
        if not state.is_current(my_id):
            raise RuntimeError("CANCELLED")
    finally:
        state.clear(my_id)
        try:
            resp.close()
        except Exception:
            pass


# 内部视觉模式 id（仅后端路由：选此项时 VLM 用 robot_toy 产品 prompt，并跑对应 engine）
VISION_MODE_ROBOT_TOY = "ssg_a_robot_toy"


def build_vlm_request(
    args: argparse.Namespace,
    question: str,
    jpeg: bytes,
    *,
    robot_toy_mode: bool = False,
) -> Dict[str, Any]:
    image_url = _image_data_url_from_jpeg_bytes(
        jpeg,
        args.vlm_max_image_side,
        args.vlm_jpeg_quality,
        args.vlm_image_format,
    )
    q = (question or "").strip()
    user_text = q if q else question
    rules = build_vlm_rules(robot_toy_mode=robot_toy_mode)
    if rules.strip():
        user_text = "%s\n\n%s" % (rules, user_text)

    messages: list = []
    system_prompt = build_vlm_system_prompt(robot_toy_mode=robot_toy_mode)
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": user_text},
            ],
        }
    )
    return {
        "model": args.vlm_model,
        "messages": messages,
        "temperature": args.vlm_temperature,
        "max_tokens": args.vlm_max_tokens,
        "repeat_penalty": args.vlm_repeat_penalty,
        # 关闭 llama-server 侧与本请求相关的 prompt/KV 前缀复用（与 --no-cache-prompt/--cache-ram 0 一致）。
        "cache_prompt": False,
        "n_cache_reuse": 0,
    }


def make_handler(
    args: argparse.Namespace,
    output: LatestJPEG,
    cv_state: CvModelState,
    boundary: bytes,
):
    class Handler(BaseHTTPRequestHandler):
        # SSE / VLM 可能数十秒才有首字节，不能用几秒级 socket timeout，否则会断连，
        # 浏览器表现为 fetch 失败（Failed to fetch）。
        timeout = None

        def setup(self) -> None:
            super().setup()
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.request.settimeout(self.timeout)

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def start_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Connection", "close")
            self.end_headers()

        def send_sse(self, payload: Dict[str, Any], event: Optional[str] = None) -> None:
            if event:
                self.wfile.write(("event: %s\n" % event).encode("utf-8"))
            data = json.dumps(payload, ensure_ascii=False)
            self.wfile.write(("data: %s\n\n" % data).encode("utf-8"))
            self.wfile.flush()

        def send_static_file(self, path: str) -> None:
            full_path = os.path.abspath(os.path.join(UI_ROOT, path))
            if not full_path.startswith(UI_ROOT + os.sep) or not os.path.isfile(full_path):
                self.send_error(404)
                return

            with open(full_path, "rb") as f:
                data = f.read()
            content_type = mimetypes.guess_type(full_path)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            route = self.path.split("?", 1)[0]
            if route == "/":
                self.send_static_file("index.html")
                return

            if route.startswith("/ui/"):
                self.send_static_file(route[len("/ui/") :])
                return

            if route == "/stats":
                stats = output.stats()
                stats.setdefault("cv_model", cv_state.get())
                self.send_json(stats)
                return

            if route == "/cv_model":
                self.send_json(cv_state.snapshot())
                return

            if route == "/cv_result.json":
                self.send_json(output.cv_payload())
                return

            if route == "/snapshot.jpg":
                jpeg = output.latest_jpeg()
                if jpeg is None:
                    self.send_error(503, "No camera frame is ready yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return

            if route != "/stream":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + boundary.decode("ascii"))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            seen_seq = 0
            prefix = b"--" + boundary + b"\r\n"
            suffix = b"\r\n"
            try:
                while True:
                    seen_seq, jpeg = output.wait_next(seen_seq, timeout=5.0)
                    if jpeg is None:
                        break
                    header = (
                        b"Content-Type: image/jpeg\r\n"
                        + b"Content-Length: "
                        + str(len(jpeg)).encode("ascii")
                        + b"\r\n\r\n"
                    )
                    self.wfile.write(prefix + header + jpeg + suffix)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                pass

        def do_POST(self) -> None:
            route = self.path.split("?", 1)[0]
            if route not in ("/ask", "/cv_model"):
                self.send_error(404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json({"error": "Invalid Content-Length"}, status=400)
                return
            if length <= 0 or length > 65536:
                self.send_json({"error": "Request body is empty or too large"}, status=400)
                return

            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                self.send_json({"error": "Request body must be JSON"}, status=400)
                return

            if route == "/cv_model":
                model_name = str(body.get("model", "")).strip() if isinstance(body, dict) else ""
                try:
                    cv_state.set(model_name)
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=400)
                    return
                self.send_json(cv_state.snapshot())
                return

            question = str(body.get("question", "")).strip() if isinstance(body, dict) else ""
            if not question:
                self.send_json({"error": "Question is required"}, status=400)
                return
            stream = bool(body.get("stream")) if isinstance(body, dict) else False
            no_cv = bool(body.get("no_cv")) if isinstance(body, dict) else False
            requested_cv_model = str(body.get("cv_model", "")).strip() if isinstance(body, dict) else ""
            if requested_cv_model:
                try:
                    cv_state.set(requested_cv_model)
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=400)
                    return

            my_id = ASK_STATE.begin()

            jpeg = output.latest_jpeg()
            if jpeg is None:
                ASK_STATE.clear(my_id)
                self.send_json({"error": "No camera frame is ready yet"}, status=503)
                return

            cv_payload = output.cv_payload()
            active_mode = cv_state.get()
            robot_toy_mode = active_mode == VISION_MODE_ROBOT_TOY
            req_body = build_vlm_request(
                args,
                question,
                jpeg,
                robot_toy_mode=robot_toy_mode,
            )
            if stream:
                self.start_sse()
                answer_parts = []
                try:
                    for piece in post_chat_completions_stream_cancelable(
                        args.vlm_base_url, req_body, args.vlm_timeout, ASK_STATE, my_id
                    ):
                        if should_stop_vlm_repeat("".join(answer_parts)):
                            break
                        answer_parts.append(piece)
                        self.send_sse({"delta": piece})
                    if not ASK_STATE.is_current(my_id):
                        return
                    answer = "".join(answer_parts)
                    if not answer:
                        self.send_sse({"error": "VLM response did not include an answer"}, event="error")
                        return
                    self.send_sse({"done": True, "cv_result": cv_payload, "stats": output.stats()}, event="done")
                except RuntimeError as exc:
                    if str(exc) == "CANCELLED" or not ASK_STATE.is_current(my_id):
                        return
                    try:
                        self.send_sse({"error": str(exc)}, event="error")
                    except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                        pass
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    ASK_STATE.begin()
                return

            try:
                raw = post_chat_completions_cancelable(
                    args.vlm_base_url, req_body, args.vlm_timeout, ASK_STATE, my_id
                )
                answer = _assistant_content_message(raw)
            except RuntimeError as exc:
                if str(exc) == "CANCELLED" or not ASK_STATE.is_current(my_id):
                    return
                self.send_json({"error": str(exc)}, status=502)
                return

            if not ASK_STATE.is_current(my_id):
                return

            if not answer:
                self.send_json({"error": "VLM response did not include an answer"}, status=502)
                return

            self.send_json({"answer": answer, "cv_result": cv_payload, "stats": output.stats()})

    return Handler


def parse_args() -> argparse.Namespace:
    services = platform_services()
    video = services.get("video", {})
    vlm = services.get("vlm", {})
    vlm_host = vlm.get("host", "127.0.0.1")
    vlm_port = vlm.get("port", 8080)
    ap = argparse.ArgumentParser(description="Realtime ssg_a camera inference over MJPEG HTTP.")
    ap.add_argument("--host", default=video.get("host", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(video.get("port", 9080)))
    ap.add_argument("--device", default=video.get("camera", "/dev/video0"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--output-fps", type=float, default=10.0)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--input-size", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument(
        "--default-cv-model",
        choices=("none", "ssg_a_robot_toy"),
        default="none",
    )
    ap.add_argument("--robot-toy-engine", default=ROBOT_TOY_ENGINE)
    ap.add_argument("--robot-toy-names", default=ROBOT_TOY_CLASSES)
    ap.add_argument("--vlm-base-url", default=f"http://{vlm_host}:{vlm_port}")
    ap.add_argument("--vlm-model", default="gpt-4o")
    ap.add_argument("--vlm-timeout", type=float, default=600.0)
    ap.add_argument(
        "--vlm-max-image-side",
        type=int,
        default=0,
        help="0=不缩放，交给视觉塔 smart_resize；>0 则最长边限制",
    )
    ap.add_argument(
        "--vlm-image-format",
        type=str,
        choices=("png", "jpeg"),
        default="png",
        help="发给 VLM 的编码：png 避免二次 JPEG（默认）",
    )
    ap.add_argument(
        "--vlm-jpeg-quality",
        type=int,
        default=95,
        help="仅当 --vlm-image-format jpeg 时有效",
    )
    ap.add_argument("--vlm-temperature", type=float, default=0.4)
    ap.add_argument("--vlm-max-tokens", type=int, default=2048)
    ap.add_argument("--vlm-repeat-penalty", type=float, default=1.15)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    frames = LatestFrame()
    output = LatestJPEG()
    cv_state = CvModelState(args)
    camera = CameraReader(args, frames)
    infer = InferenceWorker(args, frames, output, cv_state)

    camera.start()
    infer.start()

    server = MJPEGServer((args.host, args.port), make_handler(args, output, cv_state, b"ssg_a_boundary"))
    print(
        "ssg_a camera server http://%s:%d/  device=%s %dx%d@%d output_fps=%.1f"
        % (args.host, args.port, args.device, args.width, args.height, args.fps, args.output_fps),
        flush=True,
    )
    print(
        "Windows: ssh -N -L 18080:127.0.0.1:%d user@<ORIN_IP>" % args.port,
        flush=True,
    )
    print("Browser: http://127.0.0.1:18080/", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        infer.stop()
        frames.stop()
        output.stop()
        server.server_close()


if __name__ == "__main__":
    main()
