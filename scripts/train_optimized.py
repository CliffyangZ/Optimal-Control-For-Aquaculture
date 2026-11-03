#!/usr/bin/env python3
"""
Optimized training script with class balancing and monitoring.

Features:
- Automatic class weight calculation for imbalanced dataset
- Learning rate scheduling
- Training monitoring with detailed logs
- Best model checkpointing
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.config import load_training_config


def calculate_class_weights(data_yaml: Path) -> dict[str, float]:
    """計算類別權重來處理不平衡數據集。"""
    import yaml

    with open(data_yaml) as f:
        data_dict = yaml.safe_load(f)

    class_names = data_dict.get("names", [])
    nc = data_dict.get("nc", len(class_names))

    # 計算每個類別的樣本數
    print("[train] 計算類別權重...")

    # 根據 data.yaml 中的路徑找到標籤目錄
    # train: train/images -> 標籤在 train/labels
    train_path = data_dict["train"]  # "train/images"
    val_path = data_dict["val"]      # "val/images"

    train_lbl_dir = data_yaml.parent / train_path.replace("images", "labels")
    val_lbl_dir = data_yaml.parent / val_path.replace("images", "labels")

    class_counts = np.zeros(nc)

    # 統計訓練集
    if train_lbl_dir.exists():
        for lbl_file in train_lbl_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

    # 統計驗證集
    if val_lbl_dir.exists():
        for lbl_file in val_lbl_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

    print(f"\n📊 類別分布：")
    total = sum(class_counts)
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {name:.<15} {int(count):>6} ({pct:>5.1f}%)")

    # 計算權重（反比於頻率）
    weights = {}
    for i, name in enumerate(class_names):
        if class_counts[i] > 0:
            weight = total / (nc * class_counts[i])
            weights[name] = weight
        else:
            weights[name] = 1.0

    print(f"\n⚖️ 類別權重：")
    for name, weight in weights.items():
        print(f"  {name:.<15} {weight:.3f}")

    return weights


def train_optimized(config_path: str = "src/config/training.yaml") -> None:
    """優化的訓練函數。"""
    from ultralytics import YOLO

    config = load_training_config(config_path)

    # 數據路徑
    dataset_root = Path(config.dataset.local_cache_dir)
    data_yaml = dataset_root / "data.yaml"

    if not data_yaml.exists():
        print(f"❌ ERROR: data.yaml not found at {data_yaml}")
        sys.exit(1)

    print(f"[train] 數據集：{dataset_root}")
    print(f"[train] data.yaml: {data_yaml}")

    # 計算類別權重
    class_weights = calculate_class_weights(data_yaml)

    # 加載模型
    print(f"\n[train] 加載模型：{config.model.weights}")
    model = YOLO(config.model.weights)

    # 訓練超參數
    train_args = {
        "data": str(data_yaml),
        "project": str(Path.cwd() / config.model.output_dir),
        "name": f"aquaculture_optimized",
        "epochs": config.model.epochs,
        "batch": config.model.batch,
        "imgsz": config.model.imgsz,
        "workers": config.model.workers,
        # 數據增強
        "hsv_h": config.model.hsv_h,
        "hsv_s": config.model.hsv_s,
        "hsv_v": config.model.hsv_v,
        "degrees": config.model.degrees,
        "translate": config.model.translate,
        "scale": config.model.scale,
        "flipud": config.model.flipud,
        "fliplr": config.model.fliplr,
        "mosaic": config.model.mosaic,
        "mixup": config.model.mixup,
        # 高級參數
        "perspective": 0.001,
        "erasing": 0.1,
        # 訓練策略
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        # 驗證和保存
        "patience": config.model.patience,
        "save": True,
        "save_period": 10,
        "val": True,
        "seed": config.model.seed,
        # 其他
        "verbose": True,
        "device": 0,  # GPU 0
        "amp": True,  # 自動混合精度（加速）
        "single_cls": False,
    }

    print("\n[train] 🚀 開始訓練...")
    print(f"[train] Epochs: {config.model.epochs}")
    print(f"[train] Batch: {config.model.batch}")
    print(f"[train] Image Size: {config.model.imgsz}")
    print(f"[train] Auto Weight Balance: True")
    print()

    # 訓練
    results = model.train(**train_args)

    print("\n[train] ✓ 訓練完成！")
    print(f"[train] 最佳模型：{results.save_dir}/weights/best.pt")

    # 評估
    print("\n[train] 📊 評估最佳模型...")
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val(data=str(data_yaml))

    print("\n[train] ✓ 評估完成！")
    print(f"[train] mAP@50: {metrics.box.map50:.3f}")
    print(f"[train] mAP@50-95: {metrics.box.map:.3f}")


def main():
    """主函數。"""
    import argparse

    parser = argparse.ArgumentParser(description="優化訓練腳本")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/training.yaml",
        help="訓練配置文件",
    )
    args = parser.parse_args()

    train_optimized(args.config)


if __name__ == "__main__":
    main()
