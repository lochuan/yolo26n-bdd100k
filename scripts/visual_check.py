"""Render 6 random train images with converted labels for visual sanity check."""
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("datasets/bdd7")
NAMES = ["person", "rider", "car", "bus", "truck", "bicycle", "motorcycle"]
COLORS = [(60, 220, 60), (220, 60, 60), (60, 60, 220), (220, 180, 40), (160, 60, 220), (60, 220, 220), (220, 100, 160)]


def main():
    random.seed(42)
    imgs = sorted((ROOT / "images" / "train").glob("*.jpg"))
    picks = random.sample(imgs, 6)
    tiles = []
    for p in picks:
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        lab = ROOT / "labels" / "train" / (p.stem + ".txt")
        for line in lab.read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            c, cx, cy, bw, bh = int(c), float(cx), float(cy), float(bw), float(bh)
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(im, (x1, y1), (x2, y2), COLORS[c], 2)
            cv2.putText(im, NAMES[c], (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[c], 2)
        im = cv2.resize(im, (640, 360))
        tiles.append(im)
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    out = Path("visual_check.jpg")
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out, "images:", [p.name for p in picks])


if __name__ == "__main__":
    main()