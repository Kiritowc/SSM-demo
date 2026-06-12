# SSM Architecture

## 目录原则

- **代码**：`cv/`、`ets/`、`vlm/` 直接在仓库根目录，`import` 名与文件夹名一致
- **配置**：只在 `configs/`，包内不再放 `configs/`
- **产物**：`artifacts/`（权重、engine、runtime yaml、训练 runs）

## Data flow (robot_toy demo)

1. `services/video/server.py` 采集摄像头帧
2. TensorRT 引擎推理 → `/cv_result.json`
3. 用户通过 `apps/web` 提问 → `/ask` → `vlm` → llama-server
4. 训练：`ssm train cv` → `artifacts/cv/runs/` → `cv.post_deploy` → TRT engine
