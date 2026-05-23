#!/usr/bin/env python3
"""
Merge feed.yolo26 and FishTracker_v2 datasets with preprocessing.

Steps:
1. Remap feed.yolo26 classes (0 → 1, feed class in combined dataset)
2. Copy images and labels to unified structure
3. Validate annotations
4. Generate train/val split
5. Create data.yaml
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import NamedTuple

import yaml
from sklearn.model_selection import train_test_split


class DatasetConfig(NamedTuple):
    output_dir: Path
    test_split: float = 0.2
    random_state: int = 42


CLASS_NAMES = ["debris", "feed", "fish", "lumpfish"]
FEED_CLASS_ID = 1  # 'feed' is class 1 in combined dataset


def remap_label_file(label_path: Path, class_mapping: dict[int, int]) -> list[str]:
    """Read label file and remap class IDs."""
    if not label_path.exists():
        return []

    lines = label_path.read_text().strip().split('\n')
    remapped = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_class = int(parts[0])
        new_class = class_mapping.get(old_class, old_class)
        remapped.append(f"{new_class} " + " ".join(parts[1:]))
    return remapped


def merge_datasets(config: DatasetConfig) -> None:
    """Merge feed.yolo26 and FishTracker_v2 into unified dataset."""
    base_dir = Path(__file__).parent.parent / "data" / "datasets"
    feed_dir = base_dir / "feed.yolo26"
    fish_dir = base_dir / "fishtracker_v2"
    output_dir = config.output_dir

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    all_images = []

    # ══════════════════════════════════════════════════════════════════════════
    # Process feed.yolo26 (remap class 0 → 1)
    # ══════════════════════════════════════════════════════════════════════════
    print("[merge] Processing feed.yolo26...")
    feed_class_mapping = {0: FEED_CLASS_ID}  # Remap class 0 to class 1

    feed_train_imgs = list((feed_dir / "train" / "images").glob("*.jpg"))
    feed_train_imgs += list((feed_dir / "train" / "images").glob("*.png"))

    for img_path in feed_train_imgs:
        lbl_path = feed_dir / "train" / "labels" / img_path.with_suffix(".txt").name

        # Copy image
        dst_img = img_dir / f"feed_{img_path.name}"
        shutil.copy2(img_path, dst_img)

        # Remap and write label
        remapped_lines = remap_label_file(lbl_path, feed_class_mapping)
        if remapped_lines:
            dst_lbl = lbl_dir / dst_img.with_suffix(".txt").name
            dst_lbl.write_text("\n".join(remapped_lines) + "\n")

        all_images.append(dst_img.name)

    print(f"  ✓ Copied {len(feed_train_imgs)} images from feed.yolo26")

    # ══════════════════════════════════════════════════════════════════════════
    # Process FishTracker_v2 (keep original classes)
    # ══════════════════════════════════════════════════════════════════════════
    print("[merge] Processing FishTracker_v2...")
    fish_class_mapping = {i: i for i in range(len(CLASS_NAMES))}  # No remapping

    # Train split
    fish_train_imgs = list((fish_dir / "train" / "images").glob("*.jpg"))
    fish_train_imgs += list((fish_dir / "train" / "images").glob("*.png"))

    for img_path in fish_train_imgs:
        lbl_path = fish_dir / "train" / "labels" / img_path.with_suffix(".txt").name

        dst_img = img_dir / f"fish_train_{img_path.name}"
        shutil.copy2(img_path, dst_img)

        remapped_lines = remap_label_file(lbl_path, fish_class_mapping)
        if remapped_lines:
            dst_lbl = lbl_dir / dst_img.with_suffix(".txt").name
            dst_lbl.write_text("\n".join(remapped_lines) + "\n")

        all_images.append(dst_img.name)

    # Valid split
    fish_valid_imgs = list((fish_dir / "valid" / "images").glob("*.jpg"))
    fish_valid_imgs += list((fish_dir / "valid" / "images").glob("*.png"))

    for img_path in fish_valid_imgs:
        lbl_path = fish_dir / "valid" / "labels" / img_path.with_suffix(".txt").name

        dst_img = img_dir / f"fish_valid_{img_path.name}"
        shutil.copy2(img_path, dst_img)

        remapped_lines = remap_label_file(lbl_path, fish_class_mapping)
        if remapped_lines:
            dst_lbl = lbl_dir / dst_img.with_suffix(".txt").name
            dst_lbl.write_text("\n".join(remapped_lines) + "\n")

        all_images.append(dst_img.name)

    print(f"  ✓ Copied {len(fish_train_imgs)} train + {len(fish_valid_imgs)} valid from FishTracker_v2")

    # ══════════════════════════════════════════════════════════════════════════
    # Validate and split
    # ══════════════════════════════════════════════════════════════════════════
    print("[merge] Validating and splitting...")

    # Filter images that have labels
    valid_images = []
    for img_name in all_images:
        lbl_path = lbl_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            valid_images.append(img_name)

    print(f"  ✓ Valid image-label pairs: {len(valid_images)}/{len(all_images)}")

    # Train/val split
    train_imgs, val_imgs = train_test_split(
        valid_images,
        test_size=config.test_split,
        random_state=config.random_state,
    )

    # Create train/val directories
    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_lbl_dir = output_dir / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Move files to train/val
    for img_name in train_imgs:
        src_img = img_dir / img_name
        dst_img = train_img_dir / img_name
        shutil.move(str(src_img), str(dst_img))

        lbl_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        src_lbl = lbl_dir / lbl_name
        dst_lbl = train_lbl_dir / lbl_name
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))

    for img_name in val_imgs:
        src_img = img_dir / img_name
        dst_img = val_img_dir / img_name
        shutil.move(str(src_img), str(dst_img))

        lbl_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        src_lbl = lbl_dir / lbl_name
        dst_lbl = val_lbl_dir / lbl_name
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))

    # Remove empty directories
    img_dir.rmdir() if img_dir.exists() else None
    lbl_dir.rmdir() if lbl_dir.exists() else None

    print(f"  ✓ Train: {len(train_imgs)} | Val: {len(val_imgs)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Generate data.yaml
    # ══════════════════════════════════════════════════════════════════════════
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": str((output_dir / "train" / "images").relative_to(output_dir.parent.parent)),
        "val": str((output_dir / "val" / "images").relative_to(output_dir.parent.parent)),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n[merge] ✓ Merge complete!")
    print(f"  Output: {output_dir}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Classes: {', '.join(CLASS_NAMES)}")


if __name__ == "__main__":
    config = DatasetConfig(
        output_dir=Path(__file__).parent.parent / "data" / "datasets" / "combined",
        test_split=0.2,
    )
    merge_datasets(config)
