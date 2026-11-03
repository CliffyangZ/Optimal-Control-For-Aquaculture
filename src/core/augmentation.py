"""
Advanced data augmentation for aquaculture fish detection.

Uses Albumentations library to provide realistic augmentations:
- Light simulation (underwater scene)
- Motion blur (water current)
- Noise (camera noise in low light)
- Color jitter
- Spatial transforms
"""

from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_training_augmentation(img_size: int = 832) -> A.Compose:
    """
    Training augmentation pipeline with advanced transforms.

    Optimized for underwater/aquaculture scenarios:
    - Strong lighting variations (water depth, angle)
    - Motion blur (water currents, camera movement)
    - Realistic noise (underwater camera sensors)
    - Color shifts (water color filtering)
    """
    return A.Compose(
        [
            # ────────────────────────────────────────────────────────
            # 光線增強 (Lighting) - 模擬水下場景
            # ────────────────────────────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.4,     # ±40% 亮度變化
                contrast_limit=0.4,       # ±40% 對比度
                p=0.6,
            ),
            A.RandomGamma(
                gamma_limit=(70, 130),    # 伽馬值：模擬光線非線性
                p=0.4,
            ),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.4,
            ),
            A.RandomRain(p=0.1),         # 雨水效果（水的波紋）
            A.RandomFog(
                fog_coef_lower=0.3,
                fog_coef_upper=0.8,
                p=0.3,
            ),

            # ────────────────────────────────────────────────────────
            # 模糊增強 (Blur) - 模擬水流和相機移動
            # ────────────────────────────────────────────────────────
            A.GaussianBlur(
                blur_limit=(3, 9),        # 核心大小 3-9
                p=0.4,
            ),
            A.MotionBlur(
                blur_limit=9,             # 更強的運動模糊
                p=0.3,
            ),
            A.MedianBlur(
                blur_limit=5,
                p=0.1,
            ),

            # ────────────────────────────────────────────────────────
            # 噪聲增強 (Noise) - 模擬水下相機噪聲
            # ────────────────────────────────────────────────────────
            A.GaussNoise(
                var_limit=(10.0, 50.0),   # 高斯噪聲
                p=0.4,
            ),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.3,
            ),

            # ────────────────────────────────────────────────────────
            # 顏色增強 (Color) - 水的顏色過濾效果
            # ────────────────────────────────────────────────────────
            A.HueSaturationValue(
                hue_shift_limit=15,       # 色調移位
                sat_shift_limit=40,       # 飽和度（水的綠/藍色）
                val_shift_limit=30,       # 亮度
                p=0.6,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4,
            ),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.5,
            ),
            A.ChannelShuffle(p=0.1),     # 偶爾打亂顏色通道

            # ────────────────────────────────────────────────────────
            # 空間變換 (Spatial) - 不同觀看角度
            # ────────────────────────────────────────────────────────
            A.Rotate(
                limit=20,                 # ±20° 旋轉
                border_mode=cv2.BORDER_REFLECT,
                p=0.6,
            ),
            A.Perspective(
                scale=(0.05, 0.15),       # 5-15% 透視變換
                p=0.4,
            ),
            A.Affine(
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                shear=(-10, 10),
                p=0.5,
            ),
            A.Flip(p=0.5),
            A.VerticalFlip(p=0.2),

            # ────────────────────────────────────────────────────────
            # 邊界增強 (Boundary)
            # ────────────────────────────────────────────────────────
            A.CoarseDropout(
                max_holes=3,
                max_height=30,
                max_width=30,
                min_holes=1,
                min_height=10,
                min_width=10,
                p=0.3,
            ),
            A.Posterize(
                num_bits=(4, 8),
                p=0.1,
            ),

            # ────────────────────────────────────────────────────────
            # Normalization & Resize
            # ────────────────────────────────────────────────────────
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(always_apply=True),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_area=0,
            min_visibility=0.3,
        ),
    )


def get_validation_augmentation(img_size: int = 832) -> A.Compose:
    """
    Validation augmentation (minimal, only resize and normalize).
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(always_apply=True),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_area=0,
            min_visibility=0.3,
        ),
    )


class YOLOAugmentedDataset:
    """
    YOLO dataset with Albumentations augmentation.
    Compatible with PyTorch DataLoader.
    """

    def __init__(
        self,
        img_dir: str,
        lbl_dir: str,
        transform=None,
        img_size: int = 832,
    ):
        from pathlib import Path

        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.transform = transform
        self.img_size = img_size

        # Get all images
        self.images = sorted(
            list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / img_path.with_suffix(".txt").name

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels
        bboxes = []
        class_labels = []

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    class_labels.append(class_id)
                    bboxes.append(bbox)

        # Apply augmentation
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]

        return {
            "image": image,
            "bboxes": bboxes,
            "class_labels": class_labels,
            "image_path": str(img_path),
        }


if __name__ == "__main__":
    # Test augmentation
    print("Testing augmentation pipeline...")

    train_aug = get_training_augmentation(img_size=832)
    val_aug = get_validation_augmentation(img_size=832)

    print("✓ Training augmentation created")
    print("✓ Validation augmentation created")

    # List all transforms
    print("\n=== Training Augmentation Transforms ===")
    for i, transform in enumerate(train_aug.transforms):
        print(f"  {i + 1}. {transform.__class__.__name__}")
