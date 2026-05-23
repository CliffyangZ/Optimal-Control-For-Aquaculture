#!/usr/bin/env python3
"""
Data preprocessing and validation for combined dataset.

Steps:
1. Validate images (format, size, corrupted files)
2. Validate labels (format, class IDs, bbox coordinates)
3. Remove corrupted files
4. Compute class distribution
5. Generate statistics report
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class DatasetStats:
    total_images: int
    total_valid_images: int
    total_labels: int
    corrupted_images: int
    missing_labels: int
    invalid_labels: int
    class_distribution: dict[str, int]
    avg_objects_per_image: float
    bbox_stats: dict[str, dict[str, float]]


class DatasetValidator:
    """Validate and preprocess YOLO dataset."""

    def __init__(self, dataset_root: Path, class_names: list[str]):
        self.root = dataset_root
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.corrupted_images = []
        self.missing_labels = []
        self.invalid_labels = []

    def validate_image(self, img_path: Path) -> bool:
        """Check if image file is valid."""
        if not img_path.exists():
            return False

        try:
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                return False
            if img.shape[0] < 10 or img.shape[1] < 10:  # Very small images
                return False
            return True
        except Exception:
            return False

    def validate_label(self, lbl_path: Path) -> tuple[bool, list[tuple[int, float, float, float, float]]]:
        """
        Validate YOLO label file.
        Returns: (is_valid, annotations)
        """
        if not lbl_path.exists():
            return False, []

        try:
            annotations = []
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        return False, []

                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Validate class ID
                    if class_id < 0 or class_id >= self.num_classes:
                        return False, []

                    # Validate bbox coordinates (should be in [0, 1])
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                            0 < width <= 1 and 0 < height <= 1):
                        return False, []

                    annotations.append((class_id, x_center, y_center, width, height))

            return True, annotations
        except Exception:
            return False, []

    def compute_bbox_stats(self, annotations: list[tuple]) -> dict[str, dict[str, float]]:
        """Compute statistics for bounding boxes."""
        stats = {cn: {"min_w": 1.0, "max_w": 0.0, "avg_w": 0.0, "min_h": 1.0, "max_h": 0.0, "avg_h": 0.0}
                 for cn in self.class_names}

        class_widths = {cn: [] for cn in self.class_names}
        class_heights = {cn: [] for cn in self.class_names}

        for class_id, x, y, w, h in annotations:
            class_name = self.class_names[class_id]
            class_widths[class_name].append(w)
            class_heights[class_name].append(h)

        for class_name in self.class_names:
            if class_widths[class_name]:
                ws = class_widths[class_name]
                hs = class_heights[class_name]
                stats[class_name] = {
                    "min_w": float(np.min(ws)),
                    "max_w": float(np.max(ws)),
                    "avg_w": float(np.mean(ws)),
                    "min_h": float(np.min(hs)),
                    "max_h": float(np.max(hs)),
                    "avg_h": float(np.mean(hs)),
                }

        return stats

    def validate_split(self, split_name: str) -> tuple[int, int, dict, dict]:
        """Validate a dataset split (train/val)."""
        split_dir = self.root / split_name
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists():
            return 0, 0, {}, {}

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        valid_count = 0
        class_dist = {cn: 0 for cn in self.class_names}
        all_bbox_stats = {cn: [] for cn in self.class_names}

        print(f"\n[validate] Checking {split_name}...")
        for img_path in tqdm(images, desc=f"  {split_name}"):
            # Validate image
            if not self.validate_image(img_path):
                self.corrupted_images.append(str(img_path))
                continue

            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                self.missing_labels.append(str(img_path))
                # Images without labels are still valid (empty image)
                valid_count += 1
                continue

            # Validate label
            is_valid, annotations = self.validate_label(lbl_path)
            if not is_valid:
                self.invalid_labels.append(str(img_path))
                continue

            valid_count += 1
            for class_id, x, y, w, h in annotations:
                class_dist[self.class_names[class_id]] += 1
                all_bbox_stats[self.class_names[class_id]].append((x, y, w, h))

        # Compute bbox stats
        bbox_stats = {cn: {"min_w": 0, "max_w": 0, "avg_w": 0, "min_h": 0, "max_h": 0, "avg_h": 0}
                      for cn in self.class_names}
        for class_name in self.class_names:
            if all_bbox_stats[class_name]:
                bboxes = np.array(all_bbox_stats[class_name])
                bbox_stats[class_name] = {
                    "min_w": float(np.min(bboxes[:, 2])),
                    "max_w": float(np.max(bboxes[:, 2])),
                    "avg_w": float(np.mean(bboxes[:, 2])),
                    "min_h": float(np.min(bboxes[:, 3])),
                    "max_h": float(np.max(bboxes[:, 3])),
                    "avg_h": float(np.mean(bboxes[:, 3])),
                }

        return len(images), valid_count, class_dist, bbox_stats

    def generate_report(self) -> DatasetStats:
        """Generate validation report."""
        train_total, train_valid, train_classes, train_bbox = self.validate_split("train")
        val_total, val_valid, val_classes, val_bbox = self.validate_split("val")

        total_images = train_total + val_total
        total_valid = train_valid + val_valid
        total_labels = sum(train_classes.values()) + sum(val_classes.values())
        class_dist = {}

        for cn in self.class_names:
            class_dist[cn] = train_classes.get(cn, 0) + val_classes.get(cn, 0)

        avg_objects = total_labels / total_valid if total_valid > 0 else 0

        # Merge bbox stats
        bbox_stats = {}
        for cn in self.class_names:
            t_bbox = train_bbox.get(cn, {"min_w": 0, "max_w": 0, "avg_w": 0, "min_h": 0, "max_h": 0, "avg_h": 0})
            v_bbox = val_bbox.get(cn, {"min_w": 0, "max_w": 0, "avg_w": 0, "min_h": 0, "max_h": 0, "avg_h": 0})
            bbox_stats[cn] = {
                "min_w": min(t_bbox["min_w"], v_bbox["min_w"]) if (t_bbox["min_w"] > 0 or v_bbox["min_w"] > 0) else 0,
                "max_w": max(t_bbox["max_w"], v_bbox["max_w"]),
                "avg_w": (t_bbox["avg_w"] + v_bbox["avg_w"]) / 2 if (t_bbox["avg_w"] > 0 or v_bbox["avg_w"] > 0) else 0,
                "min_h": min(t_bbox["min_h"], v_bbox["min_h"]) if (t_bbox["min_h"] > 0 or v_bbox["min_h"] > 0) else 0,
                "max_h": max(t_bbox["max_h"], v_bbox["max_h"]),
                "avg_h": (t_bbox["avg_h"] + v_bbox["avg_h"]) / 2 if (t_bbox["avg_h"] > 0 or v_bbox["avg_h"] > 0) else 0,
            }

        return DatasetStats(
            total_images=total_images,
            total_valid_images=total_valid,
            total_labels=total_labels,
            corrupted_images=len(self.corrupted_images),
            missing_labels=len(self.missing_labels),
            invalid_labels=len(self.invalid_labels),
            class_distribution=class_dist,
            avg_objects_per_image=avg_objects,
            bbox_stats=bbox_stats,
        )

    def print_report(self, stats: DatasetStats) -> None:
        """Pretty-print validation report."""
        print("\n" + "=" * 70)
        print("DATASET VALIDATION REPORT")
        print("=" * 70)

        print(f"\n📊 Overall Statistics:")
        print(f"  Total images:       {stats.total_images}")
        print(f"  Valid images:       {stats.total_valid_images}")
        print(f"  Corrupted:          {stats.corrupted_images}")
        print(f"  Missing labels:     {stats.missing_labels}")
        print(f"  Invalid labels:     {stats.invalid_labels}")

        print(f"\n📦 Annotations:")
        print(f"  Total annotations:  {stats.total_labels}")
        print(f"  Avg per image:      {stats.avg_objects_per_image:.2f}")

        print(f"\n📈 Class Distribution:")
        total_objs = sum(stats.class_distribution.values())
        for class_name in self.class_names:
            count = stats.class_distribution[class_name]
            pct = (count / total_objs * 100) if total_objs > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {class_name:<12} {count:>5} ({pct:>5.1f}%) {bar}")

        print(f"\n📐 Bounding Box Statistics (normalized):")
        for class_name in self.class_names:
            stats_dict = stats.bbox_stats[class_name]
            if stats.class_distribution[class_name] > 0:
                print(f"  {class_name}:")
                print(f"    Width:  min={stats_dict['min_w']:.3f}, max={stats_dict['max_w']:.3f}, avg={stats_dict['avg_w']:.3f}")
                print(f"    Height: min={stats_dict['min_h']:.3f}, max={stats_dict['max_h']:.3f}, avg={stats_dict['avg_h']:.3f}")

        print("\n" + "=" * 70)

    def save_report(self, stats: DatasetStats, output_path: Path) -> None:
        """Save report as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(stats), f, indent=2)
        print(f"\n💾 Report saved: {output_path}")


if __name__ == "__main__":
    # Use combined dataset directly
    dataset_root = Path(__file__).parent.parent / "data" / "datasets" / "combined"

    validator = DatasetValidator(dataset_root, ["debris", "feed", "fish", "lumpfish"])
    stats = validator.generate_report()
    validator.print_report(stats)
    validator.save_report(stats, dataset_root / "validation_report.json")
