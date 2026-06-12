import os
from flask import *
from cv.cfg import TASK_FILE, TASK_HISTORY, TRAIN_LOG, trainLogDir

from .constellation import (
    DatasetIngressFacade,
    PredictionConstellationFacade,
    RuntimeCoordinate,
    TrainingObservatoryFacade,
    TrainingTaskIngressFacade,
)




app = Flask(__name__)
dataset_ingress = DatasetIngressFacade()
task_ingress = TrainingTaskIngressFacade()
observatory = TrainingObservatoryFacade()
prediction_facade = PredictionConstellationFacade()



@app.route('/sunshink/builddataset', methods=['POST'])
def build_dataset():
    """
    构建数据集
    """
    try:
        return jsonify(dataset_ingress.materialize(request.json))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/sunshink/train", methods=["POST"])
def add_tasks():
    """
    启动训练任务
    """
    if request.method == "POST":
        data = request.json
        if data:
            return jsonify(task_ingress.enqueue(data))
        else:
            return jsonify({"error": "No JSON data provided"}), 400
    else:
        return jsonify({"error": "Invalid request method"}), 405


@app.route("/sunshink/taskstatus", methods=["GET"])
def task_status():
    """
    查询训练任务状态信息
    """
    if os.path.exists(TASK_FILE):
        return jsonify(observatory.read_task_status())
    else:
        return jsonify({"error": "TASK_FILE not found"}), 404


@app.route("/sunshink/detaillog", methods=["GET"])
def query_detaillog():
    """
    查询训练任务日志
    """
    if os.path.exists(TRAIN_LOG):
        log_html = "<br>".join(observatory.read_detailed_log())
        # 返回 HTML 内容
        return render_template_string(
            f"""
        <html>
            <head>
                <title>模型训练日志详情信息</title>
            </head>
            <body>
                <h1>模型训练日志详情信息</h1>
                <pre>{log_html}</pre>
            </body>
        </html>
        """
        )
    else:
        return jsonify({"error": "Training log not found"}), 404


@app.route("/sunshink/querylog", methods=["GET"])
def query_traininglog():
    """
    查询训练任务日志
    """
    model_name = request.args.get("model_name")
    runsDir = request.args.get("run_dir")
    if runsDir is None:
        runsDir = "runs/detect/"
    if not model_name:
        return jsonify({"error": "Missing model_name parameter"}), 400
    coordinate = RuntimeCoordinate(model_name=model_name, runs_dir=runsDir)
    log_file = os.path.join(coordinate.model_dir, "train_log.txt")
    # 检查日志文件是否存在
    if os.path.exists(log_file):
        return jsonify({"log": observatory.read_model_log(coordinate)})
    else:
        return jsonify({"error": "Training log not found"}), 404


@app.route("/sunshink/historyepoch", methods=["GET"])
def model_historyepoch():
    """
    查询训练任务日志
    """
    model_name = request.args.get("model_name")
    runsDir = request.args.get("run_dir")
    if runsDir is None:
        runsDir = "runs/detect/"
    if not model_name:
        return jsonify({"error": "Missing model_name parameter"}), 400
    try:
        coordinate = RuntimeCoordinate(model_name=model_name, runs_dir=runsDir)
        historyepoch = observatory.read_history_epoch(coordinate)
        return jsonify({"historyepoch": historyepoch})
    except Exception as e:
        return jsonify({"Exception": str(e)}), 404


@app.route("/sunshink/taskshistory", methods=["GET"])
def tasks_history():
    """
    查询训练任务日志
    """
    if os.path.exists(TASK_HISTORY):
        log_html = "<br>".join(observatory.read_task_history())
        # 返回 HTML 内容
        return render_template_string(
            f"""
        <html>
            <head>
                <title>历史训练任务记录</title>
            </head>
            <body>
                <h1>历史训练任务记录</h1>
                <pre>{log_html}</pre>
            </body>
        </html>
        """
        )
    else:
        return jsonify({"error": "TASK_FILE not found"}), 404


@app.route("/sunshink/telemetry", methods=["GET"])
def telemetry_stream():
    """
    查询结构化训练生命周期事件流
    """
    model_name = request.args.get("model_name")
    runsDir = request.args.get("run_dir")
    if runsDir is None:
        runsDir = "runs/detect/"
    if not model_name:
        return jsonify({"error": "Missing model_name parameter"}), 400
    coordinate = RuntimeCoordinate(model_name=model_name, runs_dir=runsDir)
    telemetry_file = os.path.join(coordinate.model_dir, "telemetry.jsonl")
    if not os.path.exists(telemetry_file):
        return jsonify({"error": "Telemetry stream not found"}), 404
    return jsonify({"events": observatory.read_telemetry_stream(coordinate)})


@app.route("/sunshink/eventstream", methods=["GET"])
def global_event_stream():
    """
    查询全局事件总线日志
    """
    from cv.cfg import EVENT_STREAM

    if not os.path.exists(EVENT_STREAM):
        return jsonify({"error": "Event stream not found"}), 404
    return jsonify({"events": observatory.read_global_event_stream()})


@app.route("/sunshink/viewlog", methods=["GET"])
def view_traininglog():
    """
    可视化训练任务日志
    """
    model_name = request.args.get("model_name")
    runsDir = request.args.get("run_dir")
    if runsDir is None:
        runsDir = "runs/detect/"
    if not model_name:
        return jsonify({"error": "Missing model_name parameter"}), 400
    coordinate = RuntimeCoordinate(model_name=model_name, runs_dir=runsDir)
    log_file = os.path.join(coordinate.model_dir, "train_log.txt")
    if not os.path.exists(log_file):
        return jsonify({"error": "Training log not found"}), 404
    return Response(observatory.build_visual_log(coordinate), mimetype="image/png")


@app.route("/sunshink/stop", methods=["POST"])
def stop_training():
    """
    结束训练任务
    """
    if os.path.exists(trainLogDir + "task_pid.txt"):
        try:
            return jsonify(observatory.stop_training())
        except OSError as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Training task not found"}), 404


@app.route("/sunshink/predict", methods=["POST"])
def predict():
    """
    推理预测接口
    """
    if request.method == "POST":
        data = request.json.get("image")
        model_name = request.json.get("model_name")
        runsDir = request.json.get("runsDir")
        if data and model_name:
            try:
                return jsonify(prediction_facade.predict(data, model_name, runsDir))
            except FileNotFoundError as exc:
                return jsonify({"error": str(exc)}), 400
            except Exception as e:
                return jsonify({"Exception": str(e)}), 400
        else:
            return jsonify({"error": "No image data or model_name provided"}), 400
    else:
        return jsonify({"error": "Invalid request method"}), 405

