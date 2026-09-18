"""Re-quantize INT8 with Entropy (KL) calibration and a larger calibration set (fraction=0.1)."""
import shutil
from pathlib import Path

import torch

torch.backends.mkldnn.enabled = False

import onnxruntime.quantization as ortq

_orig = ortq.quantize_static


def patched(model_input, model_output, calib_reader, **kwargs):
    kwargs["calibrate_method"] = ortq.CalibrationMethod.Entropy
    print("[requant] using ENTROPY calibration (per-tensor)")
    return _orig(model_input, model_output, calib_reader, **kwargs)


ortq.quantize_static = patched

from ultralytics import YOLO

PT = "runs/detect/yolo26n_bdd7_1024/weights/best.pt"


def main():
    f8 = YOLO(PT).export(
        format="onnx",
        quantize=8,
        imgsz=1024,
        dynamic=False,
        simplify=True,
        data="configs/data_bdd7.yaml",
        split="val",
        fraction=0.1,
        batch=8,
    )
    dst = Path("models/bdd7_int8_v2.onnx")
    shutil.move(f8, dst)
    print("INT8 v2 ONNX:", dst)

    r = YOLO(str(dst), task="detect").val(data="configs/data_bdd7.yaml", imgsz=1024, batch=8, plots=False)
    per_class = {n: round(float(r.box.ap[i]), 4) for i, n in r.names.items()}
    print(f"[int8_v2] mAP50={float(r.box.map50):.4f} mAP50-95={float(r.box.map):.4f}")
    print(f"[int8_v2] per-class mAP50-95: {per_class}")


if __name__ == "__main__":
    main()