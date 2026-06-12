import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from .tool import handle_preds


class CocoMetricAssembler:
    def __init__(self, classes):
        self.classes = classes

    def build_ground_truth(self, gts):
        coco_gt = COCO()
        coco_gt.dataset = {"images": [], "annotations": []}
        annotation_id = 0
        for image_id, gt in enumerate(gts):
            for row in range(gt.shape[0]):
                annotation_id += 1
                coco_gt.dataset["images"].append({"id": image_id})
                coco_gt.dataset["annotations"].append(
                    {
                        "image_id": image_id,
                        "category_id": gt[row, 0],
                        "bbox": np.hstack([gt[row, 1:3], gt[row, 3:5] - gt[row, 1:3]]),
                        "area": np.prod(gt[row, 3:5] - gt[row, 1:3]),
                        "id": annotation_id,
                        "iscrowd": 0,
                    }
                )
        coco_gt.dataset["categories"] = [
            {"id": index, "supercategory": name, "name": name}
            for index, name in enumerate(self.classes)
        ]
        coco_gt.createIndex()
        return coco_gt

    def build_prediction(self, preds):
        coco_pred = COCO()
        coco_pred.dataset = {"images": [], "annotations": []}
        annotation_id = 0
        for image_id, pred in enumerate(preds):
            for row in range(pred.shape[0]):
                annotation_id += 1
                coco_pred.dataset["images"].append({"id": image_id})
                coco_pred.dataset["annotations"].append(
                    {
                        "image_id": image_id,
                        "category_id": int(pred[row, 0]),
                        "score": pred[row, 1],
                        "bbox": np.hstack([pred[row, 2:4], pred[row, 4:6] - pred[row, 2:4]]),
                        "area": np.prod(pred[row, 4:6] - pred[row, 2:4]),
                        "id": annotation_id,
                    }
                )
        coco_pred.dataset["categories"] = [
            {"id": index, "supercategory": name, "name": name}
            for index, name in enumerate(self.classes)
        ]
        coco_pred.createIndex()
        return coco_pred


class EvaluationTelemetryLens:
    def project(self, coco_eval):
        mAP05 = coco_eval.stats[1]
        mAP05_095 = coco_eval.stats[0]
        recall = coco_eval.stats[8]
        precision = mAP05
        f1 = 0.0 if precision + recall <= 0 else 2 * (precision * recall) / (precision + recall)
        return {
            "mAP0.5": mAP05,
            "mAP0.5:0.95": mAP05_095,
            "precision": precision,
            "recall": recall,
            "F1": f1,
        }


class CocoDetectionEvaluator:
    def __init__(self, names, device):
        self.device = device
        self.classes = []
        with open(names, "r") as file:
            for line in file.readlines():
                self.classes.append(line.strip())
        self.assembler = CocoMetricAssembler(self.classes)
        self.telemetry = EvaluationTelemetryLens()

    def coco_evaluate(self, gts, preds):
        coco_gt = self.assembler.build_ground_truth(gts)
        coco_pred = self.assembler.build_prediction(preds)
        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return self.telemetry.project(coco_eval)

    def compute_map(self, val_dataloader, model):
        gts, pts = [], []
        pbar = tqdm(val_dataloader)
        for imgs, targets in pbar:
            imgs = imgs.to(self.device).float() / 255.0
            with torch.no_grad():
                preds = model(imgs)
                output = handle_preds(preds, self.device, 0.001)

            batch_size, _, h, w = imgs.shape
            for prediction in output:
                pbboxes = []
                for box in prediction:
                    box = box.cpu().numpy()
                    score = box[4]
                    category = box[5]
                    x1, y1, x2, y2 = box[:4] * [w, h, w, h]
                    pbboxes.append([category, score, x1, y1, x2, y2])
                pts.append(np.array(pbboxes))

            for batch_index in range(batch_size):
                tbboxes = []
                for target in targets:
                    if target[0] == batch_index:
                        target = target.cpu().numpy()
                        category = target[1]
                        bcx, bcy, bw, bh = target[2:] * [w, h, w, h]
                        x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
                        x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
                        tbboxes.append([category, x1, y1, x2, y2])
                gts.append(np.array(tbboxes))
        return self.coco_evaluate(gts, pts)
