import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)

from cv.core.exporting import ExportRuntimeContext, exporter_registry


def convert_and_save_model(opt):
    if not hasattr(opt, "yaml"):
        opt.yaml = "artifacts/cv/runtime/self.yaml"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="artifacts/cv/runtime/self.yaml", help=".yaml config")
    parser.add_argument("--model", type=str, default="ssg_f", help="model type")
    parser.add_argument("--spp", type=str, default="spp", help="model type")
    parser.add_argument("--weight", type=str, default="runs/spp/ssg_f/best.bin", help=".weight config")
    parser.add_argument("--save_path", type=str, default="ssg_f.bin", help=".weight config")
    parser.add_argument("--img", type=str, default="test.jpg", help="The path of test image")
    parser.add_argument("--thresh", type=float, default=0.50, help="The path of test image")
    parser.add_argument("--ins", type=str, default=None, help=".weight config")
    parser.add_argument("--ous", type=str, default=None, help=".weight config")
    parser.add_argument(
        "--plain-onnx",
        type=str,
        default=None,
        help="Also write unencrypted ONNX for TensorRT trtexec",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip result.png preview inference after exporting",
    )
    convert_and_save_model(parser.parse_args())
