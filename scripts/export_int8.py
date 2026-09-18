"""Export trained model to FP32 ONNX and INT8 ONNX (static quantization, val-split calibration)."""
import shutil
from pathlib import Path

import torch

torch.backends.mkldnn.enabled = False  # oneDNN conv SIGFPE on EPYC 9555 (torch 2.11); export tracing runs on CPU

from ultralytics import YOLO

PT = "runs/detect/yolo26n_bdd7_1024/weights/best.pt"
OUT = Path("models")


def main():
    OUT.mkdir(exist_ok=True)
    model = YOLO(PT)

    f32 = model.export(format="onnx", imgsz=1024, dynamic=False, simplify=True)
    dst32 = OUT / "bdd7_fp32.onnx"
    shutil.move(f32, dst32)
    print("FP32 ONNX:", dst32)

    f8 = model.export(
        format="onnx",
        quantize=8,
        imgsz=1024,
        dynamic=False,
        simplify=True,
        data="configs/data_bdd7.yaml",
        split="val",
        fraction=0.05,
        batch=8,
    )
    dst8 = OUT / "bdd7_int8.onnx"
    shutil.move(f8, dst8)
    print("INT8 ONNX:", dst8)


if __name__ == "__main__":
    main()