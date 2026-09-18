"""Validate FP32-PT / FP32-ONNX / INT8-ONNX on the BDD7 val split; write comparison report."""
import json
import platform
from pathlib import Path

import torch

torch.backends.mkldnn.enabled = False  # oneDNN conv SIGFPE on EPYC 9555 (torch 2.11); CPU forward paths crash otherwise

from ultralytics import YOLO

WEIGHTS = [
    ("pt_fp32", "runs/detect/yolo26n_bdd7_1024/weights/best.pt", 0),
    ("onnx_fp32", "models/bdd7_fp32.onnx", 0),
    ("onnx_int8", "models/bdd7_int8.onnx", 0),
]


def main():
    report = {}
    for tag, w, dev in WEIGHTS:
        if not Path(w).exists():
            print(f"skip {tag}: {w} missing")
            continue
        model = YOLO(w, task="detect")
        try:
            r = model.val(data="configs/data_bdd7.yaml", imgsz=1024, device=dev, batch=16, plots=False)
        except Exception as e:
            print(f"{tag} GPU eval failed ({e}); retry CPU")
            r = model.val(data="configs/data_bdd7.yaml", imgsz=1024, device="cpu", batch=8, plots=False)
        names = r.names
        per_class = {}
        for i, name in names.items():
            per_class[name] = {
                "mAP50": round(float(r.box.ap50[i]), 4),
                "mAP50_95": round(float(r.box.ap[i]), 4),
                "precision": round(float(r.box.p[i]), 4),
                "recall": round(float(r.box.r[i]), 4),
            }
        report[tag] = {
            "weights": w,
            "mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4),
            "per_class": per_class,
        }
        print(f"[{tag}] mAP50={report[tag]['mAP50']} mAP50-95={report[tag]['mAP50_95']}")

    out = Path("models/eval_report.json")
    out.write_text(json.dumps(report, indent=2))
    print("report:", out)

    lines = ["# BDD7 INT8 vs FP32 评估报告", "",
             f"device: {platform.node()} | val: BDD100K val 10K @1024", "",
             "| model | mAP50 | mAP50-95 |", "|---|---|---|"]
    for tag, d in report.items():
        lines.append(f"| {tag} | {d['mAP50']} | {d['mAP50_95']} |")
    base = report.get("onnx_fp32")
    int8 = report.get("onnx_int8")
    if base and int8:
        lines += ["", "## INT8 vs FP32-ONNX 逐类 mAP50-95", "",
                  "| class | fp32 | int8 | delta |", "|---|---|---|---|"]
        for name in base["per_class"]:
            a = base["per_class"][name]["mAP50_95"]
            b = int8["per_class"][name]["mAP50_95"]
            lines.append(f"| {name} | {a} | {b} | {b - a:+.4f} |")
    Path("models/eval_report.md").write_text("\n".join(lines))
    print("report: models/eval_report.md")


if __name__ == "__main__":
    main()