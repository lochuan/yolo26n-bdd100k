"""QAT: quantization-aware fine-tune of best.pt, then export INT8 ONNX with embedded Q/DQ and evaluate."""
import shutil

import torch

torch.backends.mkldnn.enabled = False

from ultralytics import YOLO

PT = "runs/detect/yolo26n_bdd7_1024/weights/best.pt"


def main():
    model = YOLO(PT)
    model.train(
        data="configs/data_bdd7.yaml",
        imgsz=1024,
        epochs=8,
        quantize=8,
        lr0=1e-5,
        lrf=0.1,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=1.0,
        mosaic=0.0,
        batch=32,
        device=0,
        workers=16,
        cache=False,
        seed=0,
        name="qat_yolo26n_bdd7",
        plots=True,
        exist_ok=True,
    )

    f = YOLO("runs/detect/qat_yolo26n_bdd7/weights/best.pt").export(
        format="onnx", quantize=8, imgsz=1024, dynamic=False, simplify=True
    )
    dst = "models/bdd7_int8_qat.onnx"
    shutil.move(f, dst)
    print("QAT INT8 ONNX:", dst)

    r = YOLO(dst, task="detect").val(data="configs/data_bdd7.yaml", imgsz=1024, batch=8, plots=False)
    print(f"[qat_int8] mAP50={float(r.box.map50):.4f} mAP50-95={float(r.box.map):.4f}")
    per = {n: round(float(r.box.ap[i]), 4) for i, n in r.names.items()}
    print(f"[qat_int8] per-class mAP50-95: {per}")


if __name__ == "__main__":
    main()