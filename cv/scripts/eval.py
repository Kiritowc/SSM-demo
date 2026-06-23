#!/usr/bin/env python
"""CV validation entry point (mAP on val set)."""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("SSM_ROOT", str(_REPO))

from ssm.bootstrap import bootstrap_repo

REPO_ROOT = bootstrap_repo(_REPO)

from cv.core.backbone import MoEn
from cv.core.detector import Detector
from cv.paths import ROBOT_TOY_BEST_BIN, RUNS_DIR, RUNTIME_YAML
from cv.utils.datasets import TensorDataset, collate_fn
from cv.utils.evaluation import CocoDetectionEvaluator
from cv.utils.tool import LoadYaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cipher_suite = MoEn.load_cipher_suite()
decryptor = MoEn()
decryptor.cipher_suite = cipher_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--yaml", type=str, default=RUNTIME_YAML)
    parser.add_argument("--weight", type=str, default=ROBOT_TOY_BEST_BIN)
    parser.add_argument("--model", type=str, default="ssg_a")
    parser.add_argument("--ins", type=str, default=None)
    parser.add_argument("--ous", type=str, default=None)
    parser.add_argument("--aug", nargs="?", const=True, default=False)
    parser.add_argument("--method", type=int, default=1)
    parser.add_argument("--debug", nargs="?", const=True, default=False)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--spp", type=str, default="spp")
    parser.add_argument("--dir", type=str, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    opt = parse_args()
    if not os.path.exists(opt.yaml):
        raise SystemExit("请指定正确的配置文件路径: %s" % opt.yaml)
    if not os.path.exists(opt.weight):
        raise SystemExit("请指定正确的权重文件路径: %s" % opt.weight)

    cfg = LoadYaml(opt.yaml)
    print(cfg)
    print("load weight from:%s" % opt.weight)
    model = Detector(cfg.category_num, opt, True).to(device)

    decrypted_data = decryptor.de_model_to_memory(opt.weight)
    buffer = io.BytesIO(decrypted_data)
    state_dict = torch.load(buffer, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    evaluation = CocoDetectionEvaluator(cfg.names, device)
    val_dataset = TensorDataset(cfg.val_txt, cfg.input_width, cfg.input_height, opt)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
        drop_last=False,
        persistent_workers=True,
    )

    print("computer mAP...")
    metrics = evaluation.compute_map(val_dataloader, model)
    print("mAP0.5:", metrics["mAP0.5"])
    print("mAP0.5:0.95:", metrics["mAP0.5:0.95"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["F1"])


if __name__ == "__main__":
    main()
