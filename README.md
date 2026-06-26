# SSM

SSDet + ETS + VLM monorepo。

## 结构

```
ssdet/ ets/ vlm/     各模块：configs、数据、artifacts、scripts
apps/web/        Web 前端
apps/video/      摄像头 + 推理 + HTTP 服务（串联 SSDet/VLM）
configs/         platform.yaml（服务端口、trtexec 路径）
ssm/             CLI、路径配置、跨模块编排（video 启停等）
```

## 环境

统一使用 conda 环境 **`ssm`**（SSDet / ETS / VLM / 数据同步）：

```bash
conda activate ssm
export SSM_ROOT=$(pwd)
export PYTHONNOUSERSITE=1   # 避免 ~/.local 污染 conda torch/onnx
```

本机已配置 conda 环境 `ssm`（含 Jetson torch/cv2 及项目依赖）；脚本会自动切到该解释器。

## 启动

```bash
export SSM_ROOT=$(pwd)

make up              # 或 python -m ssm.cli up
make down            # 或 python -m ssm.cli down
```

`up` 启动 VLM + 视频服务（后台）→ http://127.0.0.1:9080/

```bash
python -m ssm.cli up --vlm-only    # 仅 VLM (:8080)
python -m ssm.cli up --video-only  # 仅视频+Web
```

本机路径若与 [`configs/platform.yaml`](configs/platform.yaml) 默认不同，可 export 覆盖：

```bash
export LLAMA_SERVER=$HOME/llama.cpp/build/bin/llama-server
export TRTEXEC=/usr/src/tensorrt/bin/trtexec
```

摄像头需用户加入 `video` 组：`sudo usermod -aG video $USER`（重新登录生效）。

VLM 模型需放在 `vlm/artifacts/models/`（`.gguf` 不入库）：

- `Qwen3VL-2B-Instruct-Q4_K_M.gguf`
- `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`

## 脚本入口

| 模块 | 命令 |
|------|------|
| SSDet 训练 | `python ssdet/scripts/train.py [--config ssdet/configs/default.yaml]` |
| SSDet 其他 | `python ssdet/scripts/{eval,infer,export_onnx,deploy}.py` |
| ETS 训练 | `python ets/scripts/train.py [--data-profile …] [--model …]` |
| ETS 其他 | `python ets/scripts/{eval,infer,export_onnx,backtest,sync_ssa}.py` |
| 训练 API | `python -m ssdet.api.api`（`train_api.enabled: true` 启用） |

## 运行时链路

摄像头 → TensorRT 推理 → `/ssdet_result.json` → Web `/ask` → VLM

训练：API 入队 → `train.py` → 成功后自动 `deploy.py --restart-camera` 生成 engine 并热加载
