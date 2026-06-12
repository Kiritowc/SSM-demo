import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)

import torch
import argparse
from torchsummary import summary
from cv.utils.tool import *
from cv.utils.datasets import *
from cv.utils.evaluation import CocoDetectionEvaluator
from cv.core.detector import Detector
from cv.core.backbone import *
# 指定后端设备CUDA&CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#解密模型文件并加载到内存
cipher_suite = MoEn.load_cipher_suite()
decryptor = MoEn()
decryptor.cipher_suite = cipher_suite




if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="artifacts/cv/runtime/self.yaml", help='.yaml config')
    parser.add_argument('--weight', type=str, default="runs/robot_toy/best.bin", help='.weight config')
    parser.add_argument('--model', type=str, default="ssg_a", help='.weight config')
    parser.add_argument('--ins', type=str, default=None, help='.weight config')
    parser.add_argument('--ous', type=str, default=None, help='.weight config')
    parser.add_argument('--aug', nargs='?', const=True, default=False, help='aug')
    parser.add_argument('--method', type=int, default=1, help="method")
    parser.add_argument('--debug', nargs='?', const=True, default=False, help='debug')
    parser.add_argument('--epochs', type=int, default=9, help="epochs")
    parser.add_argument('--delta', type=int, default=1, help="save delta")
    parser.add_argument("--spp", type=str, default='spp', help="model type")
    parser.add_argument("--dir", type=str, default='runs/', help="model type")

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
    assert os.path.exists(opt.weight), "请指定正确的权重文件路径"
    # 解析yaml配置文件
    cfg = LoadYaml(opt.yaml)    
    print(cfg) 
    # 加载模型权重
    print("load weight from:%s"%opt.weight)
    model = Detector(cfg.category_num, opt, True).to(device)

    decrypted_data = decryptor.de_model_to_memory(opt.weight)
    buffer = io.BytesIO(decrypted_data)
    state_dict = torch.load(buffer, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    # 定义验证函数
    evaluation = CocoDetectionEvaluator(cfg.names, device)
    # 数据集加载
    val_dataset = TensorDataset(cfg.val_txt, cfg.input_width, cfg.input_height, opt)
    #验证集
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg.batch_size,
                                                 shuffle=False,
                                                 collate_fn=collate_fn,
                                                 num_workers=1,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )
    # 模型评估
    print("computer mAP...")
    metrics = evaluation.compute_map(val_dataloader, model)
    print("mAP0.5:", metrics["mAP0.5"])
    print("mAP0.5:0.95:", metrics["mAP0.5:0.95"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["F1"])