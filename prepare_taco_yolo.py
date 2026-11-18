"""Convert TACO COCO annotations to YOLO detection format.

Usage:
python prepare_taco_yolo.py --images data/detection --annotations data/detection/annotations.json --output data/taco_yolo

This preserves TACO category IDs as contiguous YOLO class IDs. It does not merge categories
into the six TrashNet classes because TACO's taxonomy differs from TrashNet.
"""
from pathlib import Path
import argparse
import json
import random
import shutil


def convert(images_dir, annotation_file, output_dir, val_fraction=0.2, seed=42):
    images_dir, annotation_file, output_dir = map(Path, (images_dir, annotation_file, output_dir))
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda x: x["id"])
    cat_to_yolo = {cat["id"]: i for i, cat in enumerate(categories)}
    names = [cat["name"].replace("'", "").replace('"', "") for cat in categories]
    images = data["images"]
    annotations_by_image = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    random.seed(seed)
    image_records = images[:]
    random.shuffle(image_records)
    split_at = max(1, int(len(image_records) * (1 - val_fraction)))
    splits = {"train": image_records[:split_at], "val": image_records[split_at:]}
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = 0
    for split, records in splits.items():
        for img in records:
            source = images_dir / img["file_name"]
            if not source.exists():
                # Some TACO archives store files under a nested folder; try basename as fallback.
                source = images_dir / Path(img["file_name"]).name
            if not source.exists():
                continue
            width, height = float(img["width"]), float(img["height"])
            if width <= 0 or height <= 0:
                continue
            dest_name = f"{img['id']}_{Path(img['file_name']).name}"
            shutil.copy2(source, output_dir / "images" / split / dest_name)
            label_lines = []
            for ann in annotations_by_image.get(img["id"], []):
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0 or ann["category_id"] not in cat_to_yolo:
                    continue
                xc = (x + w / 2) / width
                yc = (y + h / 2) / height
                nw, nh = w / width, h / height
                # Clamp values to valid normalized range.
                xc, yc, nw, nh = [min(1.0, max(0.0, v)) for v in (xc, yc, nw, nh)]
                label_lines.append(f"{cat_to_yolo[ann['category_id']]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            label_path = output_dir / "labels" / split / f"{Path(dest_name).stem}.txt"
            label_path.write_text("\\n".join(label_lines), encoding="utf-8")
            copied += 1

    yaml_lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:"
    ] + [f"  {i}: {name}" for i, name in enumerate(names)]
    (output_dir / "data.yaml").write_text("\\n".join(yaml_lines) + "\\n", encoding="utf-8")
    print(f"Converted {copied} images. YOLO dataset YAML: {output_dir / 'data.yaml'}")
    print("Note: review TACO category labels and class imbalance before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder containing TACO images")
    parser.add_argument("--annotations", required=True, help="COCO instances JSON")
    parser.add_argument("--output", default="data/taco_yolo")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    convert(args.images, args.annotations, args.output, args.val_fraction)
