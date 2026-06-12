# SSM — Sunshink ML Platform

统一 monorepo：目标检测（CV）、时序建模（ETS）、千问 VLM、Web 前端与视频服务。

## 模块

| 模块 | 路径 |
|------|------|
| CV 模型 | `packages/cv/sunshink_cv/` |
| 时序模型 | `packages/ets/` |
| 千问 VLM | `packages/vlm/` + `services/vlm-server/` |
| Web | `apps/web/` |
| 视频服务 | `services/video/` |
| 配置 | `configs/` |
| 数据 | `data/` |
| 产物 | `artifacts/`（不进 git） |

## 快速开始

```bash
export SSM_ROOT=$(pwd)
cp .env.example .env
make install

# 演示
make demo
# 另开终端启动视频
python services/video/server.py

# 训练
make train-cv
make train-ets
```

## 服务端口

- VLM: `http://127.0.0.1:8080`
- 视频+Web: `http://127.0.0.1:9080`
- 训练 API（可选）: `http://127.0.0.1:5000`

详见 `docs/architecture.md`。
