# SSM

CV + ETS + VLM monorepo。

## 结构

```
cv/ ets/ vlm/     各模块：configs、数据、artifacts、scripts
apps/web/        Web 前端
apps/video/      摄像头 + 推理 + HTTP 服务（串联 CV/VLM）
configs/         platform.yaml（端口、二进制路径）
ssm/             CLI、路径配置、跨模块编排（video 启停等）
```

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

本机路径差异：

```bash
export LLAMA_SERVER=/path/to/llama-server
export TRTEXEC=/path/to/trtexec
```

VLM 模型需放在 `vlm/artifacts/models/`（`.gguf` 不入库）：

- `Qwen3VL-2B-Instruct-Q4_K_M.gguf`
- `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`

## 脚本入口

| 模块 | 命令 |
|------|------|
| CV | `python cv/scripts/{train,eval,infer,export_onnx,deploy}.py` |
| ETS | `python ets/scripts/{train,eval,infer,export_onnx,backtest}.py` |
| 训练 API | `python cv/api/api.py`（`configs/platform.yaml` 中 `train_api.enabled: true` 启用） |

## 运行时链路

摄像头 → TensorRT 推理 → `/cv_result.json` → Web `/ask` → VLM

训练：API 入队 → `train.py` → 成功后自动 `deploy.py --restart-camera` 生成 engine 并热加载
