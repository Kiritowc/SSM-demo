import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo, ensure_runtime_python

bootstrap_repo(_REPO)
ensure_runtime_python()


if __name__ == "__main__" and __package__ is None:
    __package__ = "ssdet.api"
from flask import Flask, Response, jsonify, render_template_string, request

from ssdet.cfg import DEFAULT_RUNS_DIR, EVENT_STREAM, TASK_FILE, TASK_HISTORY, TRAIN_LOG, ensure_ssdet_runtime, trainLogDir
from ssm.config import platform_services

from .handlers import DatasetHandler, ModelRun, PredictHandler, TrainEnqueue, TrainStatus
from .train_queue import start_train_queue

app = Flask(__name__)
_dataset = DatasetHandler()
_train_enqueue = TrainEnqueue()
_train_status = TrainStatus()
_predict = PredictHandler()
_queue_started = False


def _ensure_queue() -> None:
    global _queue_started
    if not _queue_started:
        ensure_ssdet_runtime()
        start_train_queue()
        _queue_started = True


@app.route("/api/v1/dataset/build", methods=["POST"])
def build_dataset():
    try:
        return jsonify(_dataset.build(request.json))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/v1/train", methods=["POST"])
def add_tasks():
    _ensure_queue()
    data = request.json
    if data:
        return jsonify(_train_enqueue.enqueue(data))
    return jsonify({"error": "No JSON data provided"}), 400


@app.route("/api/v1/task/status", methods=["GET"])
def task_status():
    if os.path.exists(TASK_FILE):
        return jsonify(_train_status.read_task_status())
    return jsonify({"error": "TASK_FILE not found"}), 404


@app.route("/api/v1/log/detail", methods=["GET"])
def query_detaillog():
    if os.path.exists(TRAIN_LOG):
        log_html = "<br>".join(_train_status.read_detailed_log())
        return render_template_string(
            "<html><body><pre>{{ log_html }}</pre></body></html>",
            log_html=log_html,
        )
    return jsonify({"error": "Training log not found"}), 404


@app.route("/api/v1/log/query", methods=["GET"])
def query_traininglog():
    model_name = request.args.get("model_name")
    runs_dir = request.args.get("run_dir") or DEFAULT_RUNS_DIR
    if not model_name:
        return jsonify({"error": "Missing model_name parameter"}), 400
    run = ModelRun(model_name=model_name, runs_dir=runs_dir)
    log_file = os.path.join(run.model_dir, "train_log.txt")
    if os.path.exists(log_file):
        return jsonify({"log": _train_status.read_model_log(run)})
    return jsonify({"error": "Training log not found"}), 404


@app.route("/api/v1/task/history-epoch", methods=["GET"])
def model_historyepoch():
    model_name = request.args.get("model_name")
    runs_dir = request.args.get("run_dir") or DEFAULT_RUNS_DIR
    if not model_name:
        return jsonify({"error": "Missing model_name parameter"}), 400
    try:
        run = ModelRun(model_name=model_name, runs_dir=runs_dir)
        return jsonify({"historyepoch": _train_status.read_history_epoch(run)})
    except Exception as exc:
        return jsonify({"Exception": str(exc)}), 404


@app.route("/api/v1/task/history", methods=["GET"])
def tasks_history():
    if os.path.exists(TASK_HISTORY):
        log_html = "<br>".join(_train_status.read_task_history())
        return render_template_string(
            "<html><body><pre>{{ log_html }}</pre></body></html>",
            log_html=log_html,
        )
    return jsonify({"error": "TASK_HISTORY not found"}), 404


@app.route("/api/v1/telemetry", methods=["GET"])
def telemetry_stream():
    model_name = request.args.get("model_name")
    runs_dir = request.args.get("run_dir") or DEFAULT_RUNS_DIR
    if not model_name:
        return jsonify({"error": "Missing model_name parameter"}), 400
    run = ModelRun(model_name=model_name, runs_dir=runs_dir)
    telemetry_file = os.path.join(run.model_dir, "telemetry.jsonl")
    if not os.path.exists(telemetry_file):
        return jsonify({"error": "Telemetry stream not found"}), 404
    return jsonify({"events": _train_status.read_telemetry_stream(run)})


@app.route("/api/v1/events", methods=["GET"])
def global_event_stream():
    if not os.path.exists(EVENT_STREAM):
        return jsonify({"error": "Event stream not found"}), 404
    return jsonify({"events": _train_status.read_global_event_stream()})


@app.route("/api/v1/log/view", methods=["GET"])
def view_traininglog():
    model_name = request.args.get("model_name")
    runs_dir = request.args.get("run_dir") or DEFAULT_RUNS_DIR
    if not model_name:
        return jsonify({"error": "Missing model_name parameter"}), 400
    run = ModelRun(model_name=model_name, runs_dir=runs_dir)
    log_file = os.path.join(run.model_dir, "train_log.txt")
    if not os.path.exists(log_file):
        return jsonify({"error": "Training log not found"}), 404
    return Response(_train_status.build_visual_log(run), mimetype="image/png")


@app.route("/api/v1/train/stop", methods=["POST"])
def stop_training():
    if os.path.exists(trainLogDir + "task_pid.txt"):
        try:
            return jsonify(_train_status.stop_training())
        except OSError as exc:
            return jsonify({"error": str(exc)}), 500
    return jsonify({"error": "Training task not found"}), 404


@app.route("/api/v1/predict", methods=["POST"])
def predict():
    body = request.json
    if not body:
        return jsonify({"error": "No JSON data provided"}), 400
    data = body.get("image")
    model_name = body.get("model_name")
    runs_dir = body.get("runsDir")
    if data and model_name:
        try:
            return jsonify(_predict.predict(data, model_name, runs_dir))
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"Exception": str(exc)}), 400
    return jsonify({"error": "No image data or model_name provided"}), 400


if __name__ == "__main__":
    _ensure_queue()
    train_api = platform_services().get("train_api", {})
    if not train_api.get("enabled", False):
        print("train_api disabled in configs/platform.yaml", flush=True)
    app.run(host="0.0.0.0", port=int(train_api.get("port", 5000)))
