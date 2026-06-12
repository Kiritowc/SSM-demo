# ETS — 工业级轻量化时序建模框架

基于 PyTorch 的时序建模工程框架，提供 LSTM/GRU/TCN/DLinear 等模型。支持单步/多步预测与时序分类，通过 YAML 配置与脚本内参数驱动。

## 功能特性

- **模型**：`LSTMModel`、`GRUModel`（`features_only` 输入）、`TCNModel`、`DLinearModel`（MS 输入：特征 + 目标历史）
- **任务类型**：时序预测（回归）与时序分类
- **预测模式**：单步（`horizon=1`）与多步（`horizon>1`）
- **训练引擎**：EarlyStopping、Checkpoint、matplotlib 可视化、对齐日志
- **推理**：批量预测，自动恢复标准化参数
- **导出**：ONNX 导出
- **工程化**：Docker、GitHub CI、分布式训练骨架

## 项目结构

```text
ets/
├── ets/              # 核心 Python 包
│   ├── models/       # BaseSequenceModel、LSTM、GRU、注册表
│   ├── data/         # 预处理、数据集、DataModule
│   ├── tasks/        # ForecastTask、ClassifyTask
│   ├── engine/       # Trainer、Evaluator、Predictor
│   ├── utils/        # 配置、日志、Checkpoint、指标
│   └── export/       # ONNX 导出
├── configs/          # YAML 配置文件
├── scripts/          # train.py、infer.py、export_onnx.py、backtest.py
├── datasets/         # 数据文件
└── docker/           # Dockerfile
```

## 快速开始

### 安装

```bash
cd e:\ets
pip install -r requirements.txt
pip install -e .
```

推荐使用 `ssdet` 环境（GPU）：

```bash
E:\conda\envs\ssdet\python.exe -m pip install -r requirements.txt
E:\conda\envs\ssdet\python.exe -m pip install -e .
```

### 训练

**直接运行**（训练超参改 `scripts/train.py` 底部 `parse_args()` 的 `default=`）：

```bash
python scripts/train.py
```

训练脚本底部 `parse_args()` 集中所有超参；`--model` 的 `choices` 自动读取模型注册表（当前 `lstm`, `gru`, `tcn`, `dlinear`），新增模型后无需改 choices：

```python
# scripts/train.py 底部 — 改 default= 即可
parser.add_argument("--model", default="lstm")  # choices 自动读取 MODEL_REGISTRY
parser.add_argument("--task", choices=["forecast", "classify"], default="forecast")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--hidden-size", default=128, type=int)
parser.add_argument("--window-size", default=24, type=int)
parser.add_argument("--monitor", default="val_rmse")  # forecast: val_rmse/val_mae；classify: val_accuracy/val_f1
```

TCN / DLinear 使用 MS 输入模式（特征 + 目标历史），需加载对应模型配置：

```bash
python scripts/train.py --model tcn --config configs/models/tcn.yaml
python scripts/train.py --model dlinear --config configs/models/dlinear.yaml
```

### 训练可视化

训练结束后，在 `runs/<实验名>_<时间戳>/` 下自动生成 matplotlib 图表：

| 文件 | 说明 |
|------|------|
| `plots/loss_curve.png` | 训练/验证 Loss 曲线 |
| `plots/metrics_curve.png` | MAE/RMSE/MAPE（或 Accuracy/F1） |
| `plots/lr_curve.png` | 学习率变化（仅在学习率实际变化时生成） |
| `plots/predictions.png` | 测试集预测对比图 |
| `history.json` | 完整训练历史（可复绘） |

训练日志会：

- 每个 epoch 先打印一行 Train/Val Loss（无时间戳）
- 每 10 个 epoch 再打印时间戳和 MAE/RMSE/MAPE 等指标（每行一个）

在 `configs/default.yaml` 中控制可视化：

```yaml
train:
  visualization:
    enabled: true
    save_dir: plots
    plot_predictions: true
    max_prediction_samples: 500
```

切换模型/任务：改 `scripts/train.py` 底部 `--model` / `--task` 的 `default=` 即可。

多步预测：改 `scripts/train.py` 底部 `--horizon` 的 `default=`。

### 推理

模型结构默认来自 **checkpoint 内 cfg**（与训练一致）。改 `scripts/infer.py` 底部 `parse_args()` 后运行：

```python
parser.add_argument("--checkpoint", default="runs/forecast_YYYYMMDD_HHMMSS")
parser.add_argument("--output", default="outputs/predictions.csv")
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--num-workers", default=0, type=int)
# 仅当需要换数据文件时开启：--merge-data-yaml
```

```bash
python scripts/infer.py
```

### ONNX 导出

完全使用 checkpoint 内 cfg。改 `scripts/export_onnx.py` 底部参数后运行：

```python
parser.add_argument("--checkpoint", default="runs/forecast_YYYYMMDD_HHMMSS")
parser.add_argument("--output", default="exports/model.onnx")
parser.add_argument("--opset-version", default=17, type=int)
parser.add_argument("--device", default="cpu")
```

```bash
python scripts/export_onnx.py
```

### Walk-forward 回测

与 `train.py` 共用模型/训练超参（底部 `parse_args()`），另加回测专用参数：

```python
parser.add_argument("--n-splits", default=5, type=int)
parser.add_argument("--fold-epochs", default=5, type=int)
parser.add_argument("--train-window", default=0, type=int)
```

```bash
python scripts/backtest.py
```

> 注意：walk-forward 回测当前仅支持 **forecast** 任务（输出 MAE/RMSE/MAPE），分类任务请用 `train.py` + `infer.py`。

## 配置说明

配置采用**分层自动合并**，顺序如下（后者覆盖前者）：

1. `configs/default.yaml` — 数据路径、特征列、可视化等
2. `configs/data/<profile>.yaml` — 按 `data.profile` 自动加载
3. `configs/models/<model.name>.yaml` — 按 `model.name` 自动加载
4. `configs/tasks/<task.type>.yaml` — 按 `task.type` 自动加载
5. `scripts/train.py` / `scripts/backtest.py` 底部 `parse_args()` — **训练超参覆盖 YAML**
6. `scripts/infer.py` / `scripts/export_onnx.py` — **入口参数在脚本底部**；模型结构以 checkpoint 为准

### 主要配置项

| 配置段 | 所在文件 | 说明 |
|--------|----------|------|
| `model.name` / `task.type` | `train.py` / `backtest.py` | 脚本底部参数覆盖 YAML |
| `model.*`（hidden_size 等） | `train.py` / `backtest.py` | 脚本底部参数覆盖 YAML |
| `data.path` / `feature_cols` | `default.yaml` | 数据集与特征列 |
| `data.window_size` / `horizon` | `train.py` / `backtest.py` | 脚本覆盖 |
| `train.epochs` / `batch_size` / `lr` | `train.py` / `backtest.py` | 脚本覆盖 |
| `train.visualization` | `default.yaml` | 画图开关与样式 |
| 推理 checkpoint / batch | `infer.py` | 模型来自 checkpoint，batch/device 在脚本改 |
| 导出 checkpoint / opset | `export_onnx.py` | 全部在脚本底部改 |
| `eval.n_splits` / `fold_epochs` | `backtest.py` | 回测专用，脚本底部改 |

### 分类标签

空气质量示例中，`Pollution_Index` 会按阈值 `[400, 700]` 划分为 3 类。在 `scripts/train.py` 底部设 `--task` 的 `default="classify"` 即可启用分类任务。

## 扩展模型

新增模型只需 **3 步**，不用为每个任务单独建配置文件：

1. **代码**：在 `ets/models/` 下实现模型，RNN 继承 `BaseSequenceModel`，其余预测模型继承 `BaseForecastModel`
2. **注册**：在 `ets/models/registry.py` 中注册：

```python
from ets.models.registry import register_model
register_model("my_model", MyModel)
```

3. **配置**：新增 `configs/models/my_model.yaml`（TCN/DLinear 等需在 `data` 段设置 `input_mode: ms`）

```yaml
model:
  name: my_model
  # 按模型填写各自超参
```

然后在 `scripts/train.py` 底部改 `--model` 的 `default=` 即可：

```python
parser.add_argument("--model", choices=[..., "my_model"], default="my_model")
```

`forecast` / `classify` 任务配置会自动沿用 `configs/tasks/` 里的统一片段，无需再建 `my_model_forecast.yaml` 这类组合文件。

## 训练产物

每次训练会在 `runs/<实验名>_<时间戳>/` 下生成：

```
runs/forecast_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt              # 验证集最优模型
│   ├── last.pt              # 最后一轮模型
│   ├── scaler.joblib        # 特征标准化参数
│   └── target_scaler.joblib # 目标标准化参数（forecast）
├── plots/                   # 训练曲线与预测图
├── history.json             # 训练历史
└── train.log                # 训练日志
```

推理时至少需要保留 `weights/` 目录（含 `best.pt` 和 `scaler.joblib`）。
断点续训：在 `scripts/train.py` 里设 `--resume` 默认值为 `runs/.../weights/last.pt`

## 数据说明

示例数据集 `datasets/New_York_Air_Quality.csv` 中含有 `#DIV/0!` 等无效值，会在 `ets/data/preprocess.py` 中于训练前自动清洗。

## Docker

```bash
docker build -t ets -f docker/Dockerfile .
docker run -v $(pwd)/runs:/app/runs ets
```

## 代码检查

```bash
ruff check ets scripts
```

## 分布式训练（预留）

分布式工具骨架位于 `ets/utils/distributed.py`。可通过 PyTorch DDP 标准环境变量（`WORLD_SIZE`、`RANK`、`LOCAL_RANK`）启动。

## 许可证

MIT
