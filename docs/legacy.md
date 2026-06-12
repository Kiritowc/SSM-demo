# Legacy path mapping

| Old | New |
|-----|-----|
| `packages/cv/sunshink_cv/` | `cv/` |
| `packages/ets/ets/` | `ets/` |
| `packages/vlm/sunshink_vlm/` | `vlm/` |
| `packages/common/ssm_common/` | `ssm_common/` |
| `python -m sunshink_cv.train` | `python -m cv.train` |
| `cv/configs/self.yaml` | `artifacts/cv/runtime/self.yaml`（由 `configs/cv/default.yaml` 生成） |
| `packages/ets/scripts/train.py` | `ets/scripts/train.py` |
