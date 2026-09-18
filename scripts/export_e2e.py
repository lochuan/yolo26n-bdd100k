"""Re-export QAT (and FP32) models with the end-to-end one-to-one head: output (batch,300,6), no external NMS."""
import shutil

import torch

torch.backends.mkldnn.enabled = False

from ultralytics import YOLO

JOBS = [
    ("runs/detect/qat_yolo26n_bdd7/weights/best.pt", "models/bdd7_int8_qat_e2e.onnx", dict(quantize=8)),
    ("runs/detect/yolo26n_bdd7_1024/weights/best.pt", "models/bdd7_fp32_e2e.onnx", dict()),
]


def main():
    for pt, dst, extra in JOBS:
        f = YOLO(pt).export(
            format="onnx",
            nms=False,
            imgsz=1024,
            dynamic=False,
            simplify=True,
            **extra,
        )
        shutil.move(f, dst)
        import onnx

        m = onnx.load(dst)
        out = m.graph.output[0]
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"E2E export: {dst} | output {out.name} {dims}")

        r = YOLO(dst, task="detect").val(data="configs/data_bdd7.yaml", imgsz=1024, batch=8, plots=False)
        print(f"[{dst}] mAP50={float(r.box.map50):.4f} mAP50-95={float(r.box.map):.4f}")
        per = {n: round(float(r.box.ap[i]), 4) for i, n in r.names.items()}
        print(f"[{dst}] per-class mAP50-95: {per}")


if __name__ == "__main__":
    main()