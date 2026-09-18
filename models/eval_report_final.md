# BDD7 YOLO26n INT8 量化最终报告

val: BDD100K val 10K 张 @1024 | 7类: person/rider/car/bus/truck/bicycle/motorcycle

## 总览

| 模型 | mAP50 | mAP50-95 | 与FP32相对差 | 体积 |
|---|---|---|---|---|
| pt_fp32 (best.pt) | 0.5967 | 0.3602 | — | 15.6 MB |
| onnx_fp32 | 0.5955 | 0.3572 | -0.2% | 9.5 MB |
| onnx_int8 (PTQ MinMax 351张) | 0.5412 | 0.3079 | -9.1% / -13.8% | 3.0 MB |
| **onnx_int8_qat (交付)** | **0.5894** | **0.3521** | **-1.0% / -1.4%** | **3.0 MB** |

## 逐类 mAP50-95: QAT INT8 vs FP32 ONNX

| class | fp32 | qat int8 | delta | 相对差 |
|---|---|---|---|---|
| person | 0.3318 | 0.3278 | -0.0040 | -1.2% |
| rider | 0.2420 | 0.2389 | -0.0031 | -1.3% |
| car | 0.5020 | 0.4995 | -0.0025 | -0.5% |
| bus | 0.4821 | 0.4778 | -0.0043 | -0.9% |
| truck | 0.4631 | 0.4542 | -0.0089 | -1.9% |
| bicycle | 0.2575 | 0.2459 | -0.0116 | -4.5% |
| motorcycle | 0.2218 | 0.2203 | -0.0015 | -0.7% |

验收标准: person 类相对掉点 <5% → 实际 -1.2% ✅ 全部类别 <5% ✅

## 结论
1. 交付模型: models/bdd7_int8_qat.onnx (QAT, 默认 one-to-many 头, 输出 [batch,11,21504], NMS 由部署侧后处理)
2. PTQ (ORT MinMax) 掉点过大 (person -15%), 已弃用; ORT Entropy 校准在 157GB RAM 下 OOM, 不可行
3. QAT (nvidia-modelopt, 8 epoch, lr0=1e-5, AdamW, cos_lr, mosaic=0) 精度保留最佳
