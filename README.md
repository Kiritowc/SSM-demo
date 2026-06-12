# SSM — Sunshink ML Platform

统一 monorepo：目标检测（CV）、时序建模（ETS）、千问 VLM、Web 前端与视频服务。

## 模块（扁平结构）

| 模块 | 路径 | import |
|------|------|--------|
| CV 模型 | `cv/` | `import cv` |
| 时序模型 | `ets/` | `import ets` |
| 千问 VLM | `vlm/` | `import vlm` |
| 平台公共 | `ssm_common/` | `import ssm_common` |
| 配置 | `configs/` | 全平台唯一配置源 |
| Web | `apps/web/` | — |
| 视频服务 | `services/video/` | — |

## 快速开始

```bash
export SSM_ROOT=$(pwd)
cp .env.example .env
make install
make smoke

# 演示
make demo
python services/video/server.py   # 另开终端

# 训练
make train-cv
make train-ets
```

## 配置说明

- **静态配置**：`configs/cv/`、`configs/ets/`、`configs/vlm/`
- **运行时配置**（训练自动生成）：`artifacts/cv/runtime/self.yaml`

## 服务端口

- VLM: `http://127.0.0.1:8080`
- 视频+Web: `http://127.0.0.1:9080`
