import sys
from pathlib import Path

if __package__ in (None, ""):
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)

import io
import cv2
import numpy as np
import argparse
import onnxruntime
from cv.core.backbone import *



dMoEn = MoEn()



class ssDet():
    def __init__(self, input_width=416, input_height=416, conf=0.3, nms=0.4, weight="", verbose=True, names="artifacts/cv/runtime/self.names"):
        self.classes = [line.strip() for line in open(names, 'r').readlines() if line.strip()]
        self.inpWidth = input_width
        self.inpHeight = input_height
        self.confThreshold = conf
        self.nmsThreshold = nms
        self.verbose = verbose
        self.H, self.W = 26, 26
        self.grid = self._make_grid(self.W, self.H)
        decrypted_data = dMoEn.de_model_to_memory(weight)
        buffer = io.BytesIO(decrypted_data)
        self.session = onnxruntime.InferenceSession(buffer.read())
        self.input_name = self.session.get_inputs()[0].name

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIds = []
        confidences = []
        boxes = []
        resData = {}
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId] * detection[0]
            if confidence > self.confThreshold:
                center_x = int(detection[1] * frameWidth)
                center_y = int(detection[2] * frameHeight)
                width = int(detection[3] * frameWidth)
                height = int(detection[4] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            box = boxes[i]
            left = max(box[0], 0)
            top = max(box[1], 0)
            width = box[2]
            height = box[3]
            one_label = self.classes[classIds[i]]
            one_score = float(confidences[i])
            one_box = [left, top, left + width, top + height]
            if self.verbose:
                print("box: ", left, top, left + width, top + height)
                print("label: ", one_label, ", score: ", one_score)
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
            if self.classes[classIds[i]] in resData:
                resData[one_label].append([one_label, one_box, one_score])
            else:
                resData[one_label] = [[one_label, one_box, one_score]]
        return resData, frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _flatten_detection_output(self, outs):
        """Turn ONNX tensor into (num_cells, num_channels). Supports NCHW / HWCH / NC."""
        outs = np.asarray(outs)
        exp_c = len(self.classes) + 5

        if outs.ndim == 4:
            if outs.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got shape {outs.shape}")
            outs = outs[0]

        if outs.ndim == 2:
            if outs.shape[1] == exp_c:
                return outs
            if outs.shape[0] == exp_c:
                return outs.T
            raise ValueError(
                f"2D output shape {outs.shape} does not match {exp_c} channels (names + 5 bbox fields)"
            )

        if outs.ndim != 3:
            raise ValueError(f"Unexpected ONNX output rank {outs.ndim}, shape {outs.shape}")

        d0, d1, d2 = outs.shape
        if d0 == exp_c:
            c, h, w = d0, d1, d2
            flat = outs.reshape(c, h * w).T
        elif d2 == exp_c:
            h, w, c = d0, d1, d2
            flat = outs.reshape(h * w, c)
        elif d1 == exp_c:
            h, c, w = d0, d1, d2
            flat = outs.transpose(1, 0, 2).reshape(c, h * w).T
        elif d1 == d2 and d0 < d1:
            c, h, w = d0, d1, d2
            flat = outs.reshape(c, h * w).T
        elif d0 == d1 and d2 < d0:
            h, w, c = d0, d1, d2
            flat = outs.reshape(h * w, c)
        else:
            raise ValueError(
                f"Cannot flatten ONNX shape {outs.shape} for expected channel count {exp_c}; "
                "check layout (NCHW vs HWC) or names vs export."
            )

        self.H, self.W = h, w
        self.grid = self._make_grid(self.W, self.H)
        return flat

    def detect(self, srcimg):
        if srcimg is None or getattr(srcimg, "size", 0) == 0:
            raise ValueError("detect() expects a non-empty BGR numpy image")
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        inputs = {self.input_name: blob}
        outs = self.session.run(None, inputs)[0]
        outs = self._flatten_detection_output(outs)
        if outs.shape[1] != len(self.classes) + 5:
            nc_out = outs.shape[1] - 5
            raise ValueError(
                f"Model outputs {outs.shape[1]} channels per cell ({nc_out} class logits), "
                f"but the loaded names file lists {len(self.classes)} classes ({len(self.classes) + 5} channels expected). "
                "Use the names file that matches this ONNX export, or re-export the model."
            )
        outs[:, 3:5] = self.sigmoid(outs[:, 3:5])
        outs[:, 1:3] = (np.tanh(outs[:, 1:3]) + self.grid) / np.tile(np.array([self.W, self.H]), (outs.shape[0], 1))
        return self.postprocess(srcimg, outs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='dataset/robot_toy/images/train/7.jpg', help="image path")
    parser.add_argument('--conf', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nms', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--weight', type=str, default="runs/robot_toy/run.bin", help='.onnx config')
    parser.add_argument(
        '--names',
        type=str,
        default=None,
        help='类别名文件路径（不传则默认 artifacts/cv/runtime/self.names）',
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='outputs/inference-result.jpg',
        help='write annotated BGR image here',
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='only print detections / run inference, do not write an image',
    )
    args = parser.parse_args()
    names_path = args.names
    if names_path is None:
        names_path = 'artifacts/cv/runtime/self.names'
    src_path = Path(args.source).expanduser()
    if not src_path.is_file():
        raise SystemExit("image not found: %s" % src_path.resolve())
    srcimg = cv2.imread(str(src_path))
    if srcimg is None or srcimg.size == 0:
        raise SystemExit("opencv could not read image (unsupported format or corrupted): %s" % src_path.resolve())
    model = ssDet(conf=args.conf, nms=args.nms, weight=args.weight, names=names_path)
    _res_data, annotated = model.detect(srcimg)
    if not args.no_save:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), annotated):
            raise SystemExit("failed to write output: %s" % out_path)
        print("wrote: %s" % out_path.resolve(), flush=True)