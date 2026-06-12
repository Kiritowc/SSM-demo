# SSM Architecture

## Data flow (robot_toy demo)

1. `services/video/server.py` 采集摄像头帧
2. TensorRT 引擎推理 → `/cv_result.json`
3. 用户通过 `apps/web` 提问 → `/ask` → `sunshink_vlm` → llama-server
4. 训练：`ssm train cv` → `artifacts/cv/runs/` → `post_deploy` → TRT engine

## Packages vs Services

- **packages/** 可 `pip install -e`，无长驻端口
- **services/** 独立进程，读 `configs/platform.yaml`
