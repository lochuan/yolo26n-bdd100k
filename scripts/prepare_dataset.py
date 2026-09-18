"""Build 7-class BDD100K subset (person/rider/car/bus/truck/bicycle/motorcycle) from raw Kaggle YOLO dataset."""
import os
import sys
from collections import Counter
from pathlib import Path

SRC = Path.home() / "yolo26-bdd-vehicle-ped" / "raw_kaggle"
DST = Path.home() / "yolo26-bdd-vehicle-ped" / "datasets" / "bdd7"

KEEP = set(range(7))
NAMES = ["person", "rider", "car", "bus", "truck", "bicycle", "motorcycle"]
DROPPED = {7: "traffic light", 8: "traffic sign", 9: "train"}


def convert_labels(split: str) -> Counter:
    src_dir = SRC / split / "labels"
    dst_dir = DST / "labels" / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    n_files = 0
    for f in src_dir.glob("*.txt"):
        lines = []
        for line in f.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cid = int(parts[0])
            if cid not in KEEP:
                counts[900 + cid] += 1
                continue
            counts[cid] += 1
            lines.append(" ".join([str(cid)] + parts[1:]))
        (dst_dir / f.name).write_text("\n".join(lines) + ("\n" if lines else ""))
        n_files += 1
    return counts, n_files


def link_images(split: str):
    link = DST / "images" / split
    link.parent.mkdir(parents=True, exist_ok=True)
    target = SRC / split / "images"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(target.resolve(), link)


def main():
    if not SRC.exists():
        sys.exit(f"source not found: {SRC}")
    for split in ("train", "val"):
        link_images(split)
        counts, n = convert_labels(split)
        print(f"[{split}] label files written: {n}")
        total = sum(v for k, v in counts.items() if k < 100)
        print(f"[{split}] kept boxes: {total} | dropped: "
              + ", ".join(f"{DROPPED[k-900]}={v}" for k, v in sorted(counts.items()) if k >= 900))
        print(f"[{split}] per-class: " + ", ".join(f"{NAMES[k]}={counts[k]}" for k in range(7)))
    yaml = f"""path: {DST}
train: images/train
val: images/val
nc: {len(NAMES)}
names: {NAMES}
"""
    (DST / "data.yaml").write_text(yaml)
    print(f"wrote {DST/'data.yaml'}")
    tr_stems = {p.stem for p in (SRC / "train" / "images").glob("*.jpg")}
    va_stems = {p.stem for p in (SRC / "val" / "images").glob("*.jpg")}
    print(f"train/val stem overlap: {len(tr_stems & va_stems)}")


if __name__ == "__main__":
    main()