import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from sunshink_cv.core.registry import NamedComponentRegistry
from .tool import LoadYaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
datasetCfg = LoadYaml(str(_REPO_ROOT / "cv/configs/self.yaml"))
input_shape = [datasetCfg.input_height, datasetCfg.input_width]

augmentation_registry = NamedComponentRegistry("augmentation")


@dataclass(frozen=True)
class AugmentationDirective:
    enabled: bool
    method: int


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    return image.convert("RGB")


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image
    return image.resize((w, h), Image.BICUBIC)


def nprand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def preprocess_input(image):
    image /= 255.0
    return image


def baseRandomAug(image, boxes, jitter=0.3, hue=0.1, sat=0.7, val=0.4, random=True):
    if isinstance(image, np.ndarray):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)
    iw, ih = image.size
    h, w = input_shape
    output = []
    for box in boxes:
        _, category = box[0], box[1]
        bx, by, bw, bh = box[2], box[3], box[4], box[5]
        x1, y1 = int((bx - 0.5 * bw) * iw), int((by - 0.5 * bh) * ih)
        x2, y2 = int((bx + 0.5 * bw) * iw), int((by + 0.5 * bh) * ih)
        output.append([x1, y1, x2, y2, category])
    box = np.array(output, dtype=float)
    if not random:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return image_data, box

    new_ar = iw / ih * nprand(1 - jitter, 1 + jitter) / nprand(1 - jitter, 1 + jitter)
    scale = nprand(0.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)
    dx = int(nprand(0, w - nw))
    dy = int(nprand(0, h - nh))
    new_image = Image.new("RGB", (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image
    flip = nprand() < 0.5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image_data = np.array(image, np.uint8)
    jitter_vector = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    hue_channel, sat_channel, val_channel = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype = image_data.dtype
    x = np.arange(0, 256, dtype=jitter_vector.dtype)
    lut_hue = ((x * jitter_vector[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * jitter_vector[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * jitter_vector[2], 0, 255).astype(dtype)
    image_data = cv2.merge(
        (
            cv2.LUT(hue_channel, lut_hue),
            cv2.LUT(sat_channel, lut_sat),
            cv2.LUT(val_channel, lut_val),
        )
    )
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2BGR)
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]
    nL = len(box)
    labels_out = np.zeros((nL, 6))
    if nL:
        box[:, [0, 2]] = box[:, [0, 2]] / input_shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] / input_shape[0]
        box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
        box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        labels_out[:, 1] = box[:, -1]
        labels_out[:, 2:] = box[:, :4]
    return image_data, labels_out


def random_scale(image, boxes):
    height, width, _ = image.shape
    cw, ch = random.randint(int(width * 0.75), width), random.randint(int(height * 0.75), height)
    cx, cy = random.randint(0, width - cw), random.randint(0, height - ch)
    roi = image[cy : cy + ch, cx : cx + cw]
    roi_h, roi_w, _ = roi.shape
    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height
        x1, y1 = bx - 0.5 * bw, by - 0.5 * bh
        x2, y2 = bx + 0.5 * bw, by + 0.5 * bh
        x1 = np.clip(x1 - cx, 0, roi_w)
        x2 = np.clip(x2 - cx, 0, roi_w)
        y1 = np.clip(y1 - cy, 0, roi_h)
        y2 = np.clip(y2 - cy, 0, roi_h)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        bx, by = x1 + 0.5 * bw, y1 + 0.5 * bh
        output.append([index, category, bx / roi_w, by / roi_h, bw / roi_w, bh / roi_h])
    if not output:
        return roi, np.zeros((0, 6), dtype=float)
    return roi, np.array(output, dtype=float)


def _augmentation_identity(image, label):
    return image, label


def _augmentation_random_scale(image, label):
    return random_scale(image, label)


def _augmentation_chromatic(image, label):
    return baseRandomAug(image, label)


augmentation_registry.register("identity", _augmentation_identity)
augmentation_registry.register("random-scale", _augmentation_random_scale)
augmentation_registry.register("chromatic-warp", _augmentation_chromatic)


def collate_fn(batch):
    img, label = zip(*batch)
    for index, one_label in enumerate(label):
        if one_label.shape[0] > 0:
            one_label[:, 0] = index
    return torch.stack(img), torch.cat(label, 0)


class DetectionSampleRepository:
    def __init__(self, manifest_path: str):
        assert os.path.exists(manifest_path), "%s文件路径错误或不存在" % manifest_path
        self.path = manifest_path
        self.img_formats = ["bmp", "jpg", "jpeg", "png"]
        self.data_list = self._collect_paths()

    def _collect_paths(self):
        data_list = []
        with open(self.path, "r") as file:
            for line in file.readlines():
                data_path = line.strip()
                if not os.path.exists(data_path):
                    raise Exception("%s is not exist" % data_path)
                img_type = data_path.split(".")[-1].lower()
                if img_type not in self.img_formats:
                    raise Exception("img type error:%s" % img_type)
                data_list.append(data_path)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class AugmentationOrchestrator:
    def __init__(self, directive: AugmentationDirective):
        self.directive = directive

    def apply(self, image, label):
        if not self.directive.enabled:
            return augmentation_registry.build("identity", image, label)
        if int(self.directive.method) == 1:
            return augmentation_registry.build("random-scale", image, label)
        return augmentation_registry.build("chromatic-warp", image, label)


class TensorDataset:
    def __init__(self, path, img_width, img_height, opt):
        self.repository = DetectionSampleRepository(path)
        self.directive = AugmentationDirective(enabled=opt.aug, method=opt.method)
        self.orchestrator = AugmentationOrchestrator(self.directive)
        self.debug = opt.debug
        self.img_width = img_width
        self.img_height = img_height

    def __getitem__(self, index):
        img_path = self.repository[index]
        label_path = self._derive_label_path(img_path)
        img = cv2.imread(img_path)
        if not os.path.exists(label_path):
            raise Exception("%s is not exist" % label_path)

        label = []
        with open(label_path, "r") as file:
            for line in file.readlines():
                one = line.strip().split(" ")
                label.append([0, one[0], one[1], one[2], one[3], one[4]])
        label = np.array(label, dtype=np.float32)
        if label.shape[0]:
            assert label.shape[1] == 6, "> 5 label columns: %s" % label_path

        img, label = self.orchestrator.apply(img, label)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        if self.debug:
            self._emit_debug_frame(img, label)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), torch.from_numpy(label)

    def _derive_label_path(self, img_path):
        swapped = img_path.replace("images", "labels")
        root, _ = os.path.splitext(swapped)
        return root + ".txt"

    def _emit_debug_frame(self, img, label):
        for box in label:
            bx, by, bw, bh = box[2], box[3], box[4], box[5]
            x1 = int((bx - 0.5 * bw) * self.img_width)
            y1 = int((by - 0.5 * bh) * self.img_height)
            x2 = int((bx + 0.5 * bw) * self.img_width)
            y2 = int((by + 0.5 * bh) * self.img_height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("debug.jpg", img)

    def __len__(self):
        return len(self.repository)
