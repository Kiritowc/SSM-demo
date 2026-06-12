import io
import json
import os

import yaml
import torch
from datetime import datetime, timedelta

from sunshink_cv.cfg import *
from sunshink_cv.core.backbone import *


def _iou_xyxy_1_nm(one: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    """one:(1,4) rest:(M,4)，返回 IoU 向量 shape (M,)。"""
    if rest.shape[0] == 0:
        return torch.empty(0, device=one.device, dtype=one.dtype)
    a = one
    tl = torch.max(a[..., :2], rest[..., :2])
    br = torch.min(a[..., 2:], rest[..., 2:])
    inter = torch.clamp(br[..., 0] - tl[..., 0], min=0) * torch.clamp(
        br[..., 1] - tl[..., 1], min=0
    )
    area_a = torch.clamp(a[..., 2] - a[..., 0], min=0) * torch.clamp(
        a[..., 3] - a[..., 1], min=0
    )
    area_b = torch.clamp(rest[..., 2] - rest[..., 0], min=0) * torch.clamp(
        rest[..., 3] - rest[..., 1], min=0
    )
    denom = area_a.expand_as(area_b) + area_b - inter + 1e-6
    return inter / denom


def _nms_pure(
    boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float
) -> torch.Tensor:
    """仅 Tensor 的 NMS，不依赖 torchvision C++ 扩展。"""
    n = boxes.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    kept = []
    while order.numel() > 0:
        idx = order[0]
        kept.append(idx)
        if order.numel() == 1:
            break
        one = boxes[idx : idx + 1]
        rest_ord = order[1:]
        rest = boxes[rest_ord]
        iou = _iou_xyxy_1_nm(one, rest)
        order = rest_ord[iou <= iou_thresh]
    return torch.stack(kept)


def _batched_nms_pure(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_thresh: float,
) -> torch.Tensor:
    """等价于 torchvision 的坐标偏移技巧 + 单次 NMS。"""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    max_c = boxes.max()
    offs = labels.to(dtype=boxes.dtype) * (max_c + boxes.new_tensor(1.0))
    shifted = boxes + offs.unsqueeze(1)
    return _nms_pure(shifted, scores, iou_thresh)


def _batched_nms_safe(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_thresh: float,
) -> torch.Tensor:
    try:
        import torchvision.ops as tv_ops

        return tv_ops.batched_nms(boxes, scores, labels, iou_thresh)
    except (RuntimeError, OSError):
        pass
    return _batched_nms_pure(boxes, scores, labels, iou_thresh)


dMoEn = MoEn()


class LoadYaml:
    def __init__(self, path):
        with open(path, encoding="utf8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.val_txt = data["DATASET"]["VAL"]
        self.train_txt = data["DATASET"]["TRAIN"]
        self.names = data["DATASET"]["NAMES"]
        self.learn_rate = data["TRAIN"]["LR"]
        self.batch_size = data["TRAIN"]["BATCH_SIZE"]
        self.milestones = data["TRAIN"]["MILESTIONES"]
        self.end_epoch = data["TRAIN"]["END_EPOCH"]
        self.input_width = data["MODEL"]["INPUT_WIDTH"]
        self.input_height = data["MODEL"]["INPUT_HEIGHT"]
        self.category_num = data["MODEL"]["NC"]


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
    """
    后处理(归一化后的坐标)
    """
    total_bboxes, output_bboxes = [], []
    # 将特征图转换为检测框的坐标
    N, C, H, W = preds.shape
    bboxes = torch.zeros((N, H, W, 6))
    pred = preds.permute(0, 2, 3, 1)
    # 前背景分类分支
    pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
    # 检测框回归分支
    preg = pred[:, :, :, 1:5]
    # 目标类别分类分支
    pcls = pred[:, :, :, 5:]
    # 检测框置信度
    bboxes[..., 4] = pobj.squeeze(-1) * pcls.max(dim=-1)[0]
    bboxes[..., 5] = pcls.argmax(dim=-1)
    # 检测框的坐标
    try:
        gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    except TypeError:
        gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W))
    bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid()
    bcx = (preg[..., 0].tanh() + gx.to(device)) / W
    bcy = (preg[..., 1].tanh() + gy.to(device)) / H
    # cx,cy,w,h = > x1,y1,x2,y1
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H * W, 6)
    total_bboxes.append(bboxes)
    batch_bboxes = torch.cat(total_bboxes, 1)
    # 对检测框进行NMS处理
    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        # 阈值筛选
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])
        # NMS（优先 torchvision；扩展损坏时改用纯 Torch 实现）
        if len(b) > 0:
            b = torch.Tensor(b).to(device)
            c = torch.Tensor(c).squeeze(1).to(device)
            s = torch.Tensor(s).squeeze(1).to(device)
            keep = _batched_nms_safe(b, s, c, nms_thresh)
            for i in map(int, keep.detach().cpu().tolist()):
                output.append(temp[i])
        output_bboxes.append(torch.Tensor(output))
    return output_bboxes


def calTimeDelta(timestamp1="09:21:22", timestamp2="10:21:22", format="%H:%M:%S"):
    """
    计算给定的两个时间之间的差值
    """
    T1 = datetime.strptime(timestamp1, format)
    T2 = datetime.strptime(timestamp2, format)
    delta = T2 - T1
    day_num = delta.days
    sec_num = delta.seconds
    sec = 86400 * day_num + sec_num
    return sec


def getFutureDay(timestamp, days, format="%Y-%m-%d"):
    """
    以给定时间戳为基准，前进 days 天得到对应的时间戳
    """
    now_time = datetime.strptime(timestamp, format)
    for i in range(days):
        now_time += timedelta(days=1)
    next_timestamp = now_time.strftime(format)
    return next_timestamp


def calculate_next_training_time(allow_train_time_list):
    """
    计算下一个训练时间
    """
    current_day = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    # current_time = "23:30:00"
    print("current_time: ", current_time)
    delta_dict = {}
    neg_flag = False
    neg_delta_dict = {}
    for one_list in allow_train_time_list:
        one_start, one_end = one_list
        one_delta = calTimeDelta(
            timestamp1=one_start, timestamp2=current_time, format="%H:%M:%S"
        )
        delta_dict[one_start] = one_delta
        if one_delta <= 0:
            neg_flag = True
            neg_delta_dict[one_start] = one_delta
    print("delta_dict: ", delta_dict)
    if neg_flag:
        neg_num = len(list(neg_delta_dict.keys()))
        if neg_num == 1:
            sorted_list = sorted(delta_dict.items(), key=lambda e: e[1])
            next_start_time = current_day + " " + sorted_list[0][0]
        else:
            sorted_list = sorted(
                neg_delta_dict.items(), key=lambda e: e[1], reverse=True
            )
            next_start_time = current_day + " " + sorted_list[0][0]
    else:
        last_day = getFutureDay(current_day, 1, format="%Y-%m-%d")
        start_list = [one[0] for one in allow_train_time_list]
        start_list.sort()
        next_start_time = last_day + " " + start_list[0]
    print("next_start_time: ", next_start_time)
    return next_start_time


def is_training_time():
    """
    检查当前时间是否在允许的训练时间段内
    """
    try:
        with open(taskCfgDir + "task_config.json") as f:
            allow_train_time_list = json.load(f)["allow_train_time_list"]
    except:
        allow_train_time_list = [["01:01:01", "23:59:59"]]
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("current_time: ", current_time)
    for period in allow_train_time_list:
        start_time = period[0]
        end_time = period[1]
        # 将时间字符串转换为 datetime.time 对象
        start_time_obj = datetime.strptime(start_time, "%H:%M:%S").time()
        end_time_obj = datetime.strptime(end_time, "%H:%M:%S").time()
        # 如果时间段跨午夜
        if start_time_obj > end_time_obj:
            # 如果当前时间在午夜之前
            if current_time >= start_time:
                return True
            # 如果当前时间在午夜之后
            if current_time <= end_time:
                return True
        else:
            # 如果当前时间在时间段内
            if start_time <= current_time <= end_time:
                return True
    return False


def save_training_state(model, optimizer, scheduler, epoch, batch_num, ema, save_dir):
    """
    保存训练状态
    """
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "batch_num": batch_num,  # 保存当前的 batch 数
        "ema_state_dict": ema.shadow,  # 保存 EMA 的影子参数
    }
    state_path = os.path.join(save_dir, "training_state.bin")
    dMoEn.en_save_model(state, state_path)
    print(f"当前训练状态信息已经成功转储: {state_path}, 等待后续继续训练......")


def load_training_state(model, optimizer, scheduler, ema, save_dir):
    """
    加载训练状态
    """
    state_path = save_dir + "training_state.bin"
    if os.path.exists(state_path):
        # 解密模型文件到内存
        ddata = dMoEn.de_model_to_memory(state_path)
        buffer = io.BytesIO(ddata)
        # 加载状态
        state = torch.load(buffer, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        epoch = state["epoch"]
        batch_num = state.get("batch_num", 0)  # 获取 batch_num，默认为 0
        ema.shadow = state["ema_state_dict"]  # 加载 EMA 的影子参数
        print(f"Training state loaded from {state_path}")
        return model, optimizer, scheduler, ema, epoch, batch_num
    else:
        print("No training state found.")
        return model, optimizer, scheduler, ema, 0, 0


def get_history_epoch(save_dir):
    """
    加载训练状态
    """
    state_path = save_dir + "training_state.bin"
    if os.path.exists(state_path):
        # 解密模型文件到内存
        ddata = dMoEn.de_model_to_memory(state_path)
        buffer = io.BytesIO(ddata)
        # 加载状态
        state = torch.load(buffer, map_location=device)
        epoch = state["epoch"]
        print("当前模型历史已累积训练Epoch数为: ", epoch)
        return epoch
    else:
        print("No training state found.")
        return 0
