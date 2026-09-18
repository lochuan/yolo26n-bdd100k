# BDD7 YOLO26n 最终评估报告 / Final Evaluation Report

val: BDD100K val 10K 张 @各分辨率 | 7类: person/rider/car/bus/truck/bicycle/motorcycle

## 总览 / Overview

| 模型 | 输入 | mAP50 | mAP50-95 | 说明 |
|---|---|---|---|---|
| FP32 best.pt | 1024×1024 | 0.5967 | 0.3602 | 原始训练 |
| QAT INT8 @1024 | 1024×1024 | 0.5894 | 0.3521 | 量化损失 -1.2% |
| FP32 @640sq | 640×640 | 0.5036 | 0.2982 | 纯分辨率参照 |
| **QAT INT8 @640×384 (交付)** | **640×384** | **0.5021** | **0.2939** | rect=True 适配训练 |

## 交付模型逐类 mAP50-95 / Per-class (INT8@640×384 vs FP32@1024)

| class | FP32@1024 | INT8@640×384 | 相对差 |
|---|---|---|---|
| person | 0.3318 | 0.2522 | -24.0% |
| rider | 0.2420 | 0.1810 | -25.2% |
| car | 0.5020 | 0.4471 | -10.9% |
| bus | 0.4821 | 0.4165 | -13.6% |
| truck | 0.4631 | 0.4034 | -12.9% |
| bicycle | 0.2575 | 0.1878 | -27.1% |
| motorcycle | 0.2218 | 0.1695 | -23.6% |

## 方法论说明 / Methodology
- @1024 数字来自 Ultralytics val;@640×384 为静态矩形输入,Ultralytics val 不支持,
  使用自研评估管线(letterbox+ORT+multi_label NMS+match_predictions 复刻+ap_per_class),
  已与 Ultralytics val 在 640 方形 ONNX 上交叉验证(0.5028 vs 0.5026,偏差 0.0002)。
- PTQ(MinMax 351张)掉点 9-15%;ORT Entropy 校准 OOM(>160GB)—— 均弃用,最终采用 QAT。
- 交付: yolo26n-bdd100k-int8-640x384.onnx | 原始权重: weights/yolo26n-bdd100k-fp32.pt
