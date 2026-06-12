#!/usr/bin/env python3
"""
Offline augmentation for YOLO-format bbox datasets (per-split doubling).

Reads ``<input>/images/{train,val}`` with matching ``<input>/labels/{train,val}``, augments each
fold (original + one augmented replica per labelled image). Writes ``<output>/images|labels/...``.

File stems use consecutive integers: train indices first, then val (globally unique across splits).

Examples::

    python -m sunshink_cv.tools.augment_yolo_detection_dataset \\
        --input-root dataset/robot_toy --output-root dataset/robot_toy_aug --clean-output

Dependencies:
    pip install albumentations opencv-python numpy
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default: legacy robot_toy -> robot_toy_aug paths (resolved under repo root).

# Set for reproducibility; change via ``--seed``.
DEFAULT_RNG_SEED = 42

# Retry stochastic augment pipeline so most images yield a valid augmented sibling.
AUG_RETRY_LIMIT = 32

IMAGE_EXTS_LOWER = {".jpg", ".jpeg"}


@dataclass
class SampleRecord:
    """One sample: disk copy (``src_path``) or in-memory BGR ndarray (``bgr``)."""

    src_path: Optional[Path]
    bgr: Optional[np.ndarray]
    boxes: List[Tuple[float, float, float, float]]

    def image_suffix(self) -> str:
        if self.src_path is not None:
            suf = self.src_path.suffix.lower()
            return suf if suf else ".jpg"
        return ".jpg"


def write_fold_records(
    records: Sequence[SampleRecord],
    images_dir: Path,
    labels_dir: Path,
    *,
    idx_start: int,
) -> int:
    """
    Writes ``records`` using consecutive stems ``idx_start``, ``idx_start+1``, …
    Returns the next unused global index (= idx_start + len(records)).
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    idx = idx_start
    for rec in records:
        ext = rec.image_suffix()
        dst_img = images_dir / ("%d%s" % (idx, ext))
        dst_lbl = labels_dir / ("%d.txt" % idx)
        if rec.bgr is not None:
            if not cv2.imwrite(str(dst_img), rec.bgr):
                raise OSError("cv2.imwrite failed: %s" % dst_img)
        elif rec.src_path is not None:
            shutil.copy2(rec.src_path, dst_img)
        else:
            raise RuntimeError("SampleRecord needs src_path or bgr")
        save_yolo_labels(dst_lbl, rec.boxes, [0] * len(rec.boxes))
        idx += 1
    return idx


def natural_image_sort_key(path: Path) -> Tuple:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), stem.lower())
    m = re.match(r"(\d+)", stem)
    if m:
        return (1, int(m.group(1)), stem.lower())
    return (2, stem.lower())


def list_input_images(directory: Path) -> List[Path]:
    imgs: List[Path] = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS_LOWER:
            imgs.append(p)
    imgs.sort(key=natural_image_sort_key)
    return imgs


def write_yolo_train_val_manifests(aug_root: Path) -> None:
    """
    Write ``train.txt`` / ``val.txt`` under *aug_root* (one image path per line,
    ``./dataset/...`` style so training can resolve from repo root).
    """
    repo_root = aug_root.resolve().parents[1]
    rel = aug_root.resolve().relative_to(repo_root)
    prefix = "./" + rel.as_posix().rstrip("/")
    for split, fname in (("train", "train.txt"), ("val", "val.txt")):
        img_dir = aug_root / "images" / split
        if not img_dir.is_dir():
            raise FileNotFoundError("missing image directory: %s" % img_dir)
        paths = list_input_images(img_dir)
        lines = [prefix + "/images/%s/%s" % (split, p.name) for p in paths]
        (aug_root / fname).write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        print(
            "manifest %s (%d lines)"
            % ((aug_root / fname).resolve(), len(lines)),
            flush=True,
        )


def read_yolo_labels(path: Path) -> Tuple[List[List[float]], List[int]]:
    """Read YOLO lines: ``class xc yc w h`` normalized. Returns bbox list + classes."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], []
    boxes: List[List[float]] = []
    labels: List[int] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            print(
                "[warn] %s:%d skip malformed line (need 5 fields): %s"
                % (path.resolve(), line_no, raw),
                file=sys.stderr,
            )
            continue
        cid = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        boxes.append([xc, yc, bw, bh])
        labels.append(cid)
    return boxes, labels


def yolo_bbox_is_valid_norm(
    xc: float, yc: float, w: float, h: float, eps: float = 1e-6
) -> bool:
    """Strictly positive area, fully contained in normalized pixel unit square."""
    if w <= eps or h <= eps:
        return False
    hh, hw = h * 0.5, w * 0.5
    xmin, ymin = xc - hw, yc - hh
    xmax, ymax = xc + hw, yc + hh
    if xmin < -eps or ymin < -eps or xmax > 1.0 + eps or ymax > 1.0 + eps:
        return False
    return True


def clip_yolo_bbox(
    xc: float, yc: float, w: float, h: float, eps: float = 1e-6
) -> Tuple[float, float, float, float] | None:
    """Clip YOLO box to [0,1]×[0,1]; return new (xc,yc,w,h) or ``None`` if empty."""
    hw, hh = w * 0.5, h * 0.5
    x1, y1 = xc - hw, yc - hh
    x2, y2 = xc + hw, yc + hh
    x1c = float(np.clip(x1, 0.0, 1.0))
    y1c = float(np.clip(y1, 0.0, 1.0))
    x2c = float(np.clip(x2, 0.0, 1.0))
    y2c = float(np.clip(y2, 0.0, 1.0))
    wc = x2c - x1c
    hc = y2c - y1c
    if wc <= eps or hc <= eps:
        return None
    return (x1c + x2c) * 0.5, (y1c + y2c) * 0.5, wc, hc


def save_yolo_labels(
    path: Path, bboxes: Sequence[Sequence[float]], class_ids: Sequence[int]
) -> None:
    lines: List[str] = []
    for (xc, yc, bw, bh), _ in zip(bboxes, class_ids):
        lines.append("0 %.8f %.8f %.8f %.8f" % (xc, yc, bw, bh))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_augmentation_pipeline() -> A.Compose:
    """Geometric + photometric detectors-friendly chain (moderate strength)."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.25,
            ),
            A.GaussNoise(p=0.25),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.08, 0.08),
                rotate=(-15, 15),
                p=0.7,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.15),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
            A.RandomShadow(p=0.12),
            A.RandomGamma(gamma_limit=(90, 110), p=0.12),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.25,
        ),
    )


def sanitize_augmented_bboxes(
    bboxes: Sequence[Sequence[float]],
) -> List[Tuple[float, float, float, float]]:
    """Clip each box to normalized image bounds; drop degenerate."""
    kept: List[Tuple[float, float, float, float]] = []
    for xc, yc, bw, bh in bboxes:
        clipped = clip_yolo_bbox(float(xc), float(yc), float(bw), float(bh))
        if clipped is None:
            continue
        fxc, fyc, fw, fh = clipped
        if yolo_bbox_is_valid_norm(fxc, fyc, fw, fh):
            kept.append((fxc, fyc, fw, fh))
    return kept


def augment_split(
    images_dir: Path,
    labels_dir: Path,
    transform: A.Compose,
    fold_name: str,
) -> Tuple[List[SampleRecord], int, int, int, int]:
    """Build ``originals + augmentation`` queue for train or val. Returns pooled records."""
    pairs: List[Tuple[Path, Path, List[List[float]], List[int]]] = []
    skipped_no_label = 0
    all_sorted = list_input_images(images_dir)
    raw_count = len(all_sorted)

    for img_path in all_sorted:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.is_file():
            skipped_no_label += 1
            print(
                "[skip][%s] no label for image: %s (expected %s)"
                % (fold_name, img_path.resolve(), label_path.resolve())
            )
            continue
        boxes, _lbl = read_yolo_labels(label_path)
        if not boxes:
            skipped_no_label += 1
            print(
                "[skip][%s] empty or invalid labels: %s"
                % (fold_name, label_path.resolve())
            )
            continue
        pairs.append((img_path, label_path, boxes, _lbl))

    records: List[SampleRecord] = []
    for img_path, _lp, boxes, _lbls in pairs:
        box_tpl = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in boxes]
        records.append(SampleRecord(src_path=img_path, bgr=None, boxes=box_tpl))

    skipped_augmentation = 0
    aug_saved = 0

    for img_path, _lp, boxes, _lbls in pairs:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None or img_bgr.size == 0:
            skipped_augmentation += 1
            print(
                "[skip][%s] cannot decode image with OpenCV: %s"
                % (fold_name, img_path.resolve())
            )
            continue

        zero_labels = [0] * len(boxes)
        saved_this = False
        last_exc = None

        for _attempt in range(AUG_RETRY_LIMIT):
            try:
                out = transform(image=img_bgr, bboxes=list(boxes), class_labels=zero_labels)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

            aug_boxes = sanitize_augmented_bboxes(out["bboxes"])
            if not aug_boxes:
                continue

            records.append(SampleRecord(src_path=None, bgr=out["image"], boxes=aug_boxes))
            aug_saved += 1
            saved_this = True
            break

        if not saved_this:
            skipped_augmentation += 1
            if last_exc:
                print(
                    "[skip][%s] augment failed after retries %s: %s"
                    % (fold_name, img_path.resolve(), last_exc)
                )
            else:
                print(
                    "[skip][%s] augmented bbox degenerate/out of bounds: %s"
                    % (fold_name, img_path.resolve())
                )

    return records, raw_count, skipped_no_label, skipped_augmentation, aug_saved


def resolve_under_repo(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline YOLO detection dataset augmentation.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "robot_toy",
        help="Dataset root with images/train, labels/train, images/val, labels/val.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "robot_toy_aug",
        help="Augmented dataset output root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help="RNG seed for albumentations + numpy (default: %(default)s).",
    )
    wg = parser.add_mutually_exclusive_group()
    wg.add_argument(
        "--clean-output",
        dest="clean_output",
        action="store_true",
        help="Remove output-root before generating (default).",
    )
    wg.add_argument(
        "--no-clean-output",
        dest="clean_output",
        action="store_false",
        help="Retain existing output-root (risk of leftover files mixing in).",
    )
    parser.set_defaults(clean_output=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_input_root = resolve_under_repo(args.input_root)
    output_root = resolve_under_repo(args.output_root)

    input_train_images_dir = dataset_input_root / "images" / "train"
    input_train_labels_dir = dataset_input_root / "labels" / "train"
    input_val_images_dir = dataset_input_root / "images" / "val"
    input_val_labels_dir = dataset_input_root / "labels" / "val"

    dirs_to_check = [
        ("train images", input_train_images_dir),
        ("train labels", input_train_labels_dir),
        ("val images", input_val_images_dir),
        ("val labels", input_val_labels_dir),
    ]
    for label, path in dirs_to_check:
        if not path.is_dir():
            raise SystemExit(
                "%s directory missing — not guessing:\n  %s" % (label, path.resolve())
            )

    rng = int(args.seed)
    random.seed(rng)
    np.random.seed(rng)

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)

    transform = build_augmentation_pipeline()

    train_records, raw_tr, snl_tr, saug_tr, aug_tr_saved = augment_split(
        input_train_images_dir,
        input_train_labels_dir,
        transform,
        "train",
    )
    val_records, raw_va, snl_va, saug_va, aug_va_saved = augment_split(
        input_val_images_dir,
        input_val_labels_dir,
        transform,
        "val",
    )

    global_idx = 1
    global_idx = write_fold_records(
        train_records,
        output_root / "images" / "train",
        output_root / "labels" / "train",
        idx_start=global_idx,
    )
    global_idx = write_fold_records(
        val_records,
        output_root / "images" / "val",
        output_root / "labels" / "val",
        idx_start=global_idx,
    )

    total_originals_scanned = raw_tr + raw_va
    total_pool = len(train_records) + len(val_records)
    total_skip_labels = snl_tr + snl_va
    total_skip_aug = saug_tr + saug_va
    total_augment_written = aug_tr_saved + aug_va_saved

    print("--- 增强统计 ---")
    print(
        "输入: %s\n输出: %s"
        % (dataset_input_root.resolve(), output_root.resolve())
    )
    print(
        "原始图片数量 (train(%d)+val(%d)): %d"
        % (raw_tr, raw_va, total_originals_scanned)
    )
    print("训练集拆分: 写入样本 %d (含原图+增强成功)" % len(train_records))
    print(
        "  成功拷贝原图条目: %d, 增强成功: %d"
        % (len(train_records) - aug_tr_saved, aug_tr_saved)
    )
    print("验证集拆分: 写入样本 %d (含原图+增强成功)" % len(val_records))
    print(
        "  成功拷贝原图条目: %d, 增强成功: %d"
        % (len(val_records) - aug_va_saved, aug_va_saved)
    )
    print(
        "文件名全局连续编号（不同 split 不重号）：train 为 1〜%d，val 为 %d〜%d"
        % (
            len(train_records),
            len(train_records) + 1 if val_records else len(train_records),
            global_idx - 1,
        )
    )
    print("成功复制原图条目合计: %d" % ((len(train_records) - aug_tr_saved) + (len(val_records) - aug_va_saved)))
    print("成功生成增强样本数量: %d" % total_augment_written)
    print(
        "跳过数量: %d"
        % (total_skip_labels + total_skip_aug)
    )
    print("  （无同名 txt 或空标注）: %d" % total_skip_labels)
    print(
        "  （读图失败 / 增强异常 / bbox 无效 / 写入失败）: %d"
        % total_skip_aug
    )
    print(
        "最终输出样本总量 (train+val): %d （每名有效标注原图为 原图 + 成功增强 两条）"
        % total_pool
    )
    write_yolo_train_val_manifests(output_root)
    cls_src = dataset_input_root / "labels" / "classes.txt"
    if cls_src.is_file():
        shutil.copy2(cls_src, output_root / "labels" / "classes.txt")
        print(
            "classes.txt -> %s" % ((output_root / "labels" / "classes.txt").resolve()),
            flush=True,
        )


if __name__ == "__main__":
    main()

