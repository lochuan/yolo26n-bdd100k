"""Train YOLO26n on 7-class BDD100K subset at 1024px."""
import torch

torch.backends.mkldnn.enabled = False  # oneDNN conv SIGFPE on EPYC 9555 (torch 2.11); training runs on CUDA

from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")
    model.train(
        data="configs/data_bdd7.yaml",
        imgsz=1024,
        epochs=100,
        patience=30,
        batch=64,
        device=0,
        workers=16,
        cache=False,
        amp=True,
        seed=0,
        name="yolo26n_bdd7_1024",
        plots=True,
        exist_ok=True,
    )

if __name__ == "__main__":
    main()