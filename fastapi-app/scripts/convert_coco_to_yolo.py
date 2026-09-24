"""PKU-Market-PCB: COCO → YOLOv8 格式转换脚本"""
import json
from pathlib import Path

DATASET_ROOT = Path(r"D:\smart-agent\fastapi-app\data\datasets\OmniData--PKU-Market-PCB\snapshots\master")
ANNO_DIR = DATASET_ROOT / "pcb_cocoanno"
OUTPUT_DIR = DATASET_ROOT / "labels"

CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def convert(json_path: Path, split_name: str):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    # 为每张图创建空标签(负样本)
    for img_id, fname in id_to_file.items():
        lp = OUTPUT_DIR / split_name / f"{Path(fname).stem}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.touch()

    count = 0
    for ann in coco["annotations"]:
        cat = ann["category_id"]
        if cat not in CLASS_MAP:
            continue
        yolo_cls = CLASS_MAP[cat]
        fname = id_to_file.get(ann["image_id"])
        if not fname:
            continue
        w, h = id_to_size[ann["image_id"]]
        x, y, bw, bh = ann["bbox"]
        xc = max(0, min(1, (x + bw / 2) / w))
        yc = max(0, min(1, (y + bh / 2) / h))
        nw = max(0, min(1, bw / w))
        nh = max(0, min(1, bh / h))
        lp = OUTPUT_DIR / split_name / f"{Path(fname).stem}.txt"
        with open(lp, "a") as f:
            f.write(f"{yolo_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
        count += 1

    total = len(list((OUTPUT_DIR / split_name).glob("*.txt")))
    non_empty = sum(1 for f in (OUTPUT_DIR / split_name).glob("*.txt") if f.stat().st_size > 0)
    print(f"✅ [{split_name}] {count} 条标注 | {total} 张图 | {non_empty} 有标注 | {total-non_empty} 负样本")


if __name__ == "__main__":
    print("🔄 COCO → YOLOv8 转换开始...")
    for split, jf in [("train", "train.json"), ("val", "val.json")]:
        jp = ANNO_DIR / jf
        if jp.exists():
            convert(jp, split)
        else:
            print(f"❌ 未找到 {jp}")
    print("🎉 转换完成!")