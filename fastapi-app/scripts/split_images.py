"""根据COCO JSON将图片分配到 train/val 子目录"""
import json, shutil
from pathlib import Path

DATASET_ROOT = Path(r"D:\smart-agent\fastapi-app\data\datasets\OmniData--PKU-Market-PCB\snapshots\master")
IMAGES_DIR = DATASET_ROOT / "images"
ANNO_DIR = DATASET_ROOT / "pcb_cocoanno"

for split, jf in [("train", "train.json"), ("val", "val.json")]:
    with open(ANNO_DIR / jf, "r") as f:
        coco = json.load(f)

    dst_dir = IMAGES_DIR / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img in coco["images"]:
        src = IMAGES_DIR / img["file_name"]
        dst = dst_dir / img["file_name"]
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    print(f"✅ [{split}] 复制 {copied} 张图片到 {dst_dir}")

print("🎉 图片目录整理完成!")