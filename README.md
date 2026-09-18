# YOLO26n-BDD7-INT8

**YOLO26 Nano · BDD100K 7 类道路目标检测 · QAT INT8 量化 · 跨平台 ONNX**

中文说明 | [English](#english)

---

## 中文

### 模型简介

基于 **Ultralytics YOLO26 Nano**(2.4M 参数)在 **BDD100K** 数据集上训练的 7 类道路目标检测模型,并通过 **量化感知训练(QAT)** 导出 INT8 ONNX。面向行人/车辆检测的边缘侧部署场景(QCOM Snapdragon QNN、Jetson、x86 ONNX Runtime 等)。

**7 个类别**:`person`(行人)、`rider`(骑手)、`car`、`bus`、`truck`、`bicycle`、`motorcycle`

> 从 BDD100K 原始 10 类中剔除了 traffic light / traffic sign / train,专注"在路上的人和车"。

### 性能(BDD100K val 10,000 张 @1024)

| 模型 | mAP50 | mAP50-95 | 相对 FP32 差距 | 文件体积 |
|---|---|---|---|---|
| FP32 训练权重 (best.pt) | 0.5967 | 0.3602 | — | 15.6 MB |
| FP32 ONNX 基线 | 0.5955 | 0.3572 | -0.2% | 9.5 MB |
| ~~PTQ INT8(已弃用)~~ | 0.5412 | 0.3079 | -9.1% / -13.8% | 3.0 MB |
| **QAT INT8(交付)** | **0.5894** | **0.3521** | **-1.0% / -1.4%** | 9.9 MB* |

\* QDQ 表示(FP32 权重 + 量化范围节点),推理运行时自动折叠为真 INT8 内核。

**逐类 mAP50-95(QAT INT8 vs FP32 ONNX)**:

| 类别 | FP32 | QAT INT8 | 相对差 |
|---|---|---|---|
| person | 0.3318 | 0.3278 | **-1.2%** |
| rider | 0.2420 | 0.2389 | -1.3% |
| car | 0.5020 | 0.4995 | -0.5% |
| bus | 0.4821 | 0.4778 | -0.9% |
| truck | 0.4631 | 0.4542 | -1.9% |
| bicycle | 0.2575 | 0.2459 | -4.5% |
| motorcycle | 0.2218 | 0.2203 | -0.7% |

✅ 验收标准:person 类相对掉点 < 5% —— 全部类别达标。

### 训练过程

**数据**
- BDD100K(经 Kaggle 镜像 `a7madmostafa/bdd100k-yolo`,官方 70K/10K split)
- 7 类过滤重映射:train 862,007 框 / val 123,757 框(剔除交通灯/标志/火车约 426K 框)
- 源图 1280×720,训练分辨率 **1024**(高分辨率对小目标 recall 提升显著)

**硬件**
- GPU:NVIDIA **RTX PRO 6000 Blackwell Server Edition**(96GB GDDR7,sm_120)
- CPU:AMD EPYC 9555(28 vCPU VM)/ 内存 157GB / Ubuntu 24.04

**Phase 1 — FP32 训练(约 5.5 小时)**
- `yolo26n.pt` COCO 预训练微调,imgsz=1024,batch=64,AMP,100 epochs(patience=30,实际收敛于 ~epoch 91)
- 约 213.5 秒/epoch;最终 mAP50 **0.5967** / mAP50-95 **0.3602**(P 0.711 / R 0.532)

**Phase 2 — INT8 量化感知训练 QAT(约 50 分钟)**

为什么不用 PTQ?我们实测了 PTQ 的两条路,均不可行:
1. ONNX Runtime **MinMax 静态量化**(351 张校准图):整体 mAP50 掉 9.1%,person 类掉 15.4% —— 校准集小 + MinMax 对激活离群值太敏感
2. ONNX Runtime **Entropy(KL)校准**:校准器内存爆炸(>160GB),OOM 被杀,不可用

最终采用 **Ultralytics 原生 QAT**(底层为 nvidia-modelopt):

```python
model = YOLO("best.pt")          # FP32 微调产物
model.train(
    data="configs/data_bdd7.yaml",
    quantize=8,                   # 开启 QAT:prepare_qat 用 train split 做初始范围校准,训练全程模拟 INT8
    epochs=8, lr0=1e-5, optimizer="AdamW",
    cos_lr=True, warmup_epochs=1.0, mosaic=0.0, batch=32, imgsz=1024,
)
model.export(format="onnx", quantize=8)  # QAT 导出内嵌 Q/DQ 节点,无需再做后训练校准
```

- 训练 forward 全程带 fake-quant,模型直接学会适应 INT8 数值域
- 8 epoch(约 7 分钟/epoch)后,量化模型 mAP50-95 从 PTQ 的 0.3079 恢复到 **0.3521**
- 导出的 ONNX 内嵌 308 个 Q/DQ 量化范围节点,ONNX Runtime / QNN 加载时折叠为真 INT8 内核

### 模型特点

- 🎯 **7 类专注**:只检道路参与者和行人,类间混淆少,适合安防/ADAS 下游逻辑
- 📐 **1024px 高分辨率训练**:小目标(远处骑行者/摩托车)recall 明显优于 640px 方案
- ⚡ **INT8 QAT**:整模型精度损失仅 1.0-1.4%,量化后边缘推理吞吐提升约 2-3 倍
- 🔀 **双头架构**:默认导出 one-to-many 头,输出 `[batch, 11, 21504]`,NMS 由部署侧完成(≤300 框 numpy NMS 约 2ms,可忽略);需要零后处理时可加 `nms=False` 导出端到端头 `(batch, 300, 6)`
- 📦 **跨平台**:标准 ONNX QDQ 表示,ONNX Runtime / QNN(SNPE)/ TensorRT 均可直接消费
- 🪶 **Nano 体积**:2.4M 参数,CPU/GPU/NPU 友好

### 适用场景

- **ADAS / 行车预警**:前向行人、骑行者、车辆碰撞目标检测
- **行车记录仪智能分析**:端侧实时检测(day/night 全场景,BDD100K 覆盖雨天/夜间)
- **交通监控**:车流统计、违停/占道检测、非机动车识别
- **低速无人车 / 机器人 / 无人机**:轻量化感知,3MB 级真 INT8 运行时占用
- **边缘 AI 盒子**:QCOM Snapdragon(QNN/SNPE)、NVIDIA Jetson、x86 工控机

### 快速开始

```python
from ultralytics import YOLO

model = YOLO("models/yolo26n-bdd7-int8-qat.onnx")   # ORT 自动将 QDQ 折叠为 INT8 内核
results = model.predict("street.jpg", imgsz=1024, conf=0.25)

for det in results[0].boxes:
    cls = results[0].names[int(det.cls)]            # person / rider / car / bus / truck / bicycle / motorcycle
    conf = float(det.conf)
    xyxy = det.xyxy[0].tolist()                     # NMS 在部署侧后处理完成(conf/iou 可调)
```

> 输出为 one-to-many 头 `[batch, 11, 21504]`,Ultralytics 内置后处理即含 NMS;自研后处理时对 21504 个候选做 conf 过滤 + NMS(≤300 框,约 2ms)。

### 仓库结构

```
├── models/
│   ├── yolo26n-bdd7-fp32.onnx        # FP32 ONNX 基线 (9.5 MB)
│   ├── yolo26n-bdd7-int8-qat.onnx    # 交付模型:QAT INT8 QDQ @1024 (9.9 MB)
│   ├── yolo26n-bdd7-int8-qat-384x640.onnx  # 客户端部署版:QAT INT8 @384x640 矩形输入 (9.6 MB)
│   └── eval_report_final.md          # 完整评估报告
├── weights/
│   └── yolo26n-bdd7-int8-qat.pt      # QAT 权重(可重新导出任意格式)
├── scripts/                          # 数据转换 / 训练 / QAT / 导出 / 评估全流程脚本
└── configs/data_bdd7.yaml
```

> **384×640 部署版说明**:`imgsz=384,640` 矩形输入(适配 16:9 行车画面),导出命令等价于
> `yolo export model=weights/yolo26n-bdd7-int8-qat.pt format=onnx quantize=8 imgsz=384,640 data=configs/data_bdd7.yaml`
> (8.4.x 中 `int8=True` 已更名为 `quantize=8`)。模型在 1024px 训练,384×640 推理存在小目标(20-40m 骑手/摩托)召回损失,详见性能实测。

### 环境备注

- PyTorch 2.11.0+cu128 / Ultralytics 8.4.155 / nvidia-modelopt 0.46.1 / onnxruntime
- ⚠️ AMD EPYC 9555 + torch 2.11 组合需 `torch.backends.mkldnn.enabled = False`(oneDNN conv 内核 SIGFPE,已在所有脚本内置)

### 许可

- 模型权重与代码基于 Ultralytics YOLO26,**AGPL-3.0**(商用需 Ultralytics 商业授权)
- BDD100K 数据集使用遵循 Berkeley 官方条款

---

<a name="english"></a>

## English

### Overview

A 7-class road-object detector built on **Ultralytics YOLO26 Nano** (2.4M params) trained on **BDD100K**, then quantized to INT8 via **Quantization-Aware Training (QAT)** and exported as cross-platform ONNX. Designed for edge deployment of pedestrian/vehicle detection (QCOM Snapdragon QNN, Jetson, x86 ONNX Runtime, etc.).

**Classes**: `person`, `rider`, `car`, `bus`, `truck`, `bicycle`, `motorcycle`

> Traffic light / traffic sign / train were dropped from BDD100K's original 10 classes to focus on road users and pedestrians.

### Performance (BDD100K val 10,000 images @1024)

| Model | mAP50 | mAP50-95 | Gap vs FP32 | Size |
|---|---|---|---|---|
| FP32 weights (best.pt) | 0.5967 | 0.3602 | — | 15.6 MB |
| FP32 ONNX baseline | 0.5955 | 0.3572 | -0.2% | 9.5 MB |
| ~~PTQ INT8 (deprecated)~~ | 0.5412 | 0.3079 | -9.1% / -13.8% | 3.0 MB |
| **QAT INT8 (shipped)** | **0.5894** | **0.3521** | **-1.0% / -1.4%** | 9.9 MB* |

\* QDQ representation (FP32 weights + quantization-range nodes); the runtime folds these into true INT8 kernels.

**Per-class mAP50-95 (QAT INT8 vs FP32 ONNX)**: person -1.2%, rider -1.3%, car -0.5%, bus -0.9%, truck -1.9%, bicycle -4.5%, motorcycle -0.7% — all classes within the 5% acceptance budget (person: **-1.2%**).

### Training

**Data** — BDD100K via Kaggle mirror `a7madmostafa/bdd100k-yolo` (official 70K/10K split); 862,007 train boxes / 123,757 val boxes across 7 classes after remapping (~426K boxes of traffic light/sign/train dropped); source 1280×720, trained at **1024 px**.

**Hardware** — NVIDIA **RTX PRO 6000 Blackwell Server Edition** (96 GB, sm_120), AMD EPYC 9555 (28-vCPU VM), 157 GB RAM, Ubuntu 24.04.

**Phase 1 — FP32 training (~5.5 h)**: COCO-pretrained `yolo26n.pt` fine-tuned at imgsz=1024, batch=64, AMP, 100 epochs (patience=30, converged ~epoch 91) at ~213.5 s/epoch. Final mAP50 **0.5967** / mAP50-95 **0.3602**.

**Phase 2 — INT8 QAT (~50 min)**: PTQ was ruled out empirically — ORT MinMax static quantization (351 calibration images) cost -9.1% overall / -15.4% on person; ORT Entropy calibration OOM-killed (>160 GB RAM). Instead we used **Ultralytics native QAT** (nvidia-modelopt): `model.train(quantize=8, epochs=8, lr0=1e-5, optimizer="AdamW", cos_lr=True, mosaic=0.0, batch=32)` runs fake-quantization in every forward pass after an initial range calibration on the train split. Exporting with `quantize=8` embeds 308 Q/DQ nodes (quantization ranges) directly in the ONNX graph — no post-training calibration needed. The quantized model recovered mAP50-95 from 0.3079 (PTQ) to **0.3521**.

### Key Features

- 🎯 Focused 7-class road-user detection
- 📐 High-resolution (1024 px) training for small-object recall
- ⚡ QAT INT8: only 1.0-1.4% overall accuracy loss, 2-3x edge inference speedup
- 🔀 Dual-head YOLO26: default export is one-to-many `[batch, 11, 21504]` with external NMS (~2 ms numpy NMS for ≤300 boxes); an end-to-end head `(batch, 300, 6)` is available via `nms=False` export
- 📦 Standard ONNX QDQ — consumable by ONNX Runtime / QNN (SNPE) / TensorRT
- 🪶 Nano footprint (2.4M params)

### Use Cases

Forward ADAS collision targets, dashcam edge analytics (day/night/rain), traffic monitoring and counting, low-speed autonomous vehicles / robots / drones, edge AI boxes (Snapdragon QNN, Jetson, x86).

### Quick Start

```python
from ultralytics import YOLO

model = YOLO("models/yolo26n-bdd7-int8-qat.onnx")   # ORT folds QDQ into INT8 kernels
results = model.predict("street.jpg", imgsz=1024, conf=0.25)
```

### License

Weights & code are derived from Ultralytics YOLO26 (**AGPL-3.0**; commercial use requires an Ultralytics license). Dataset usage follows the official BDD100K terms (Berkeley).
