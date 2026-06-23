import io
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import onnx
import torch
from onnxsim import simplify

from cv.core.detector import Detector
from cv.core.registry import NamedComponentRegistry
from cv.utils.tool import LoadYaml, MoEn, handle_preds


exporter_registry = NamedComponentRegistry("exporter")


@dataclass(frozen=True)
class ExportRuntimeContext:
    yaml_path: str
    model_name: str
    weight_path: str
    save_path: str
    image_path: str
    threshold: float
    spp: str
    ins: str = None
    ous: str = None
    plain_onnx_path: Optional[str] = None
    preview: bool = True


class EncryptedOnnxExporter:
    def __init__(self):
        self.vault = MoEn()

    def export(self, runtime: ExportRuntimeContext):
        assert os.path.exists(runtime.yaml_path), "请指定正确的配置文件路径"
        assert os.path.exists(runtime.weight_path), "请指定正确的模型路径"
        assert os.path.exists(runtime.image_path), "请指定正确的测试图像路径"

        device = torch.device("cpu")
        cfg = LoadYaml(runtime.yaml_path)
        opt = type(
            "ExportOpt",
            (),
            {
                "model": runtime.model_name,
                "weight": runtime.weight_path,
                "spp": runtime.spp,
                "ins": runtime.ins,
                "ous": runtime.ous,
            },
        )()
        model = Detector(cfg.category_num, opt, True).to(device)
        model.load_state_dict(self._decrypt_state_dict(runtime.weight_path))
        model.eval()

        ori_img = cv2.imread(runtime.image_path)
        res_img = cv2.resize(
            ori_img,
            (cfg.input_width, cfg.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0

        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            img,
            buffer,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
        onnx_model_data = buffer.getvalue()
        model_simp, check = simplify(onnx.load(io.BytesIO(onnx_model_data)))
        assert check, "Simplified ONNX model could not be validated"
        if runtime.plain_onnx_path:
            onnx_dir = os.path.dirname(os.path.abspath(runtime.plain_onnx_path))
            if onnx_dir:
                os.makedirs(onnx_dir, exist_ok=True)
            onnx.save(model_simp, runtime.plain_onnx_path)
            print("Wrote plain ONNX:", runtime.plain_onnx_path, flush=True)
        self.vault.en_save_model_buffer(model_simp.SerializeToString(), runtime.save_path)
        if not runtime.preview:
            return

        start = time.perf_counter()
        preds = model(img)
        end = time.perf_counter()
        inference_time = (end - start) * 1000.0
        print(f"forward time: {inference_time:.2f}ms")
        output = handle_preds(preds, device, runtime.threshold)

        label_names = []
        with open(cfg.names, "r") as file:
            for line in file.readlines():
                label_names.append(line.strip())

        h, w, _ = ori_img.shape
        for box in output[0]:
            box = box.tolist()
            obj_score = box[4]
            category = label_names[int(box[5])]
            x1, y1 = int(box[0] * w), int(box[1] * h)
            x2, y2 = int(box[2] * w), int(box[3] * h)
            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(ori_img, "%.2f" % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        cv2.imwrite("result.png", ori_img)

    def _decrypt_state_dict(self, weight_path):
        dedata = self.vault.de_model_to_memory(weight_path)
        buffer = io.BytesIO(dedata)
        return torch.load(buffer, map_location="cpu")


def convert_and_save_model(opt) -> None:
    from cv.paths import RUNTIME_YAML

    if not hasattr(opt, "yaml"):
        opt.yaml = RUNTIME_YAML
    if not hasattr(opt, "thresh"):
        opt.thresh = 0.50
    if not hasattr(opt, "spp"):
        opt.spp = "spp"
    if not hasattr(opt, "ins"):
        opt.ins = None
    if not hasattr(opt, "ous"):
        opt.ous = None

    runtime = ExportRuntimeContext(
        yaml_path=opt.yaml,
        model_name=opt.model,
        weight_path=opt.weight,
        save_path=opt.save_path,
        image_path=opt.img,
        threshold=opt.thresh,
        spp=opt.spp,
        ins=opt.ins,
        ous=opt.ous,
        plain_onnx_path=getattr(opt, "plain_onnx", None),
        preview=not getattr(opt, "skip_preview", False),
    )
    exporter_registry.build("encrypted-onnx").export(runtime)


exporter_registry.register("encrypted-onnx", lambda: EncryptedOnnxExporter())
