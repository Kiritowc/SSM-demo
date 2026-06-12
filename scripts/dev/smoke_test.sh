#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export SSM_ROOT="$ROOT"
PY="${SSM_PYTHON:-/home/sunshink/miniconda3/envs/ssdet/bin/python}"

echo "== ssm_common =="
$PY -c "from ssm_common.paths import repo_root; assert str(repo_root()) == '$ROOT'; print('root OK')"

echo "== sunshink_cv =="
$PY -c "from sunshink_cv.cfg import task_config; assert task_config['model_name']=='ssg_a'; print('cv OK', task_config['cfg_yaml']['DATASET']['TRAIN'])"

echo "== sunshink_vlm =="
$PY -c "from sunshink_vlm.prompts import build_vlm_system_prompt; p=build_vlm_system_prompt(robot_toy_mode=True); assert len(p)>10; print('vlm OK')"

echo "== ets =="
$PY -c "from ets import __version__; print('ets OK', __version__)"

echo "smoke test passed"
