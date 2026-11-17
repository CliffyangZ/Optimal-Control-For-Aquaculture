# 數據集合併和預處理指南

## 概述

已成功合併 **feed.yolo26** 和 **FishTracker_v2** 兩個數據集成為統一的 YOLO 訓練數據集。

### 數據統計

| 指標 | 數值 |
|-----|------|
| 總圖片數 | 499 |
| 訓練集 | 399 (80%) |
| 驗證集 | 100 (20%) |
| 總注解數 | 24,801 |
| 平均每圖注解數 | 49.70 |

### 類別分布

| 類別 | 數量 | 百分比 | 備註 |
|-----|------|--------|------|
| feed | 20,697 | 83.5% | 主要來自 feed.yolo26 |
| fish | 3,540 | 14.3% | FishTracker_v2 |
| debris | 477 | 1.9% | FishTracker_v2 |
| lumpfish | 87 | 0.4% | FishTracker_v2 |

## 數據位置

```
Data/datasets/combined/
├── train/
│   ├── images/     (399 images)
│   └── labels/     (399 labels)
├── val/
│   ├── images/     (100 images)
│   └── labels/     (100 labels)
└── data.yaml       (訓練配置)
```

## 使用方法

### 1. 訓練模型

```bash
# 使用合併的數據集訓練
python train.py download  # 不需要，已有本地數據集

# 直接訓練（使用 combined dataset）
python -c "
from ultralytics import YOLO
model = YOLO('yolo26s.pt')
results = model.train(
    data='Data/datasets/combined/data.yaml',
    epochs=100,
    imgsz=832,
    batch=16,
    device=0
)
"
```

或修改 Depolyment/src/config/training.yaml：

```yaml
dataset:
  roboflow_workspace: jtelectric
  roboflow_project: fishtracker_v2
  roboflow_version: 1
  roboflow_format: yolo26
  local_cache_dir: Data/datasets/combined  # ← 改這裡
```

### 2. 評估數據集

查看驗證報告：

```bash
cat Data/datasets/combined/validation_report.json
```

## 重要發現

### ⚠️ 類別不平衡

**feed 類占 83.5%，嚴重不平衡！**

建議在訓練時使用：
- **類別權重** (class_weights)：自動平衡類別損失
- **數據增強**：增加小類別的樣本變異
- **過採樣**：增加少數類別的樣本

### 📊 邊框大小分析

| 類別 | 寬度範圍 | 高度範圍 |
|-----|---------|---------|
| debris | 0.002-0.055 | 0.002-0.056 |
| feed | 0.001-0.077 | 0.002-0.098 |
| fish | 0.002-0.617 | 0.003-0.972 |
| lumpfish | 0.035-0.276 | 0.037-0.564 |

**發現**：
- `feed` 和 `debris` 都很小（平均 0.6-0.8% 的圖片）
- `fish` 和 `lumpfish` 較大，尤其 fish 有極端情況
- 需要多尺度檢測能力

## 數據品質檢查結果

✅ **通過驗證**：
- 0 個損壞圖片
- 0 個缺少標籤的圖片  
- 0 個無效標籤
- 所有邊框坐標在 [0, 1] 範圍內
- 所有類別 ID 有效

## 預處理的改動

### feed.yolo26 → combined

1. **類別重映射**：`class 0` (feed) → `class 1`
   - 原因：FishTracker_v2 的 feed 是 class 1
   
2. **文件名前綴**：`feed_` 前綴避免重名衝突

3. **分割**：300 張圖片加入 train/val 集

### FishTracker_v2 → combined

1. **文件名前綴**：`fish_train_` / `fish_valid_` 避免衝突

2. **類別保留**：保持原始類別 ID（0-3）

3. **分割保留**：Train/val 分割不變

## 訓練建議

### 1. 處理類別不平衡

在訓練前設置類別權重：

```python
from ultralytics import YOLO
import torch

model = YOLO('yolo26s.pt')

# 計算類別權重（反比於類別頻率）
class_counts = [477, 20697, 3540, 87]
class_weights = torch.tensor([sum(class_counts) / c for c in class_counts])
class_weights = class_weights / class_weights.sum() * len(class_weights)

results = model.train(
    data='Data/datasets/combined/data.yaml',
    epochs=100,
    imgsz=832,
    batch=16,
    device=0,
    # 通過 Ultralytics 的自動加權
    auto_weight=True  # 自動計算類別權重
)
```

### 2. 數據增強參數

建議在 Depolyment/src/config/training.yaml 中使用更激進的增強：

```yaml
model:
  mosaic: 1.0          # 馬賽克增強（推薦 1.0）
  mixup: 0.2           # 混合增強
  hsv_h: 0.015         # 色調變化
  hsv_s: 0.7           # 飽和度變化  
  hsv_v: 0.4           # 亮度變化
  degrees: 10          # 旋轉角度
  translate: 0.1       # 平移
  scale: 0.5           # 縮放
  flipud: 0.1          # 垂直翻轉
  fliplr: 0.5          # 水平翻轉
  close_mosaic: 10     # 最後 10 個 epoch 關閉 mosaic
```

### 3. 多尺度訓練

由於有小對象（feed, debris）和大對象（fish），建議：

```python
model.train(
    data='Data/datasets/combined/data.yaml',
    epochs=100,
    imgsz=832,     # 較大的輸入尺寸可以更好檢測小對象
    device=0,
    amp=True       # 自動混合精度加速
)
```

## 後續步驟

1. ✅ 合併數據集
2. ✅ 驗證數據品質
3. ⬜ 訓練新模型
4. ⬜ 評估性能
5. ⬜ 調整類別權重（如有必要）

## 文件列表

| 文件 | 描述 |
|-----|------|
| `scripts/merge_datasets.py` | 合併 feed.yolo26 + FishTracker_v2 的主腳本 |
| `scripts/preprocess_dataset.py` | 數據驗證和統計分析 |
| `Data/datasets/combined/data.yaml` | YOLO 訓練配置檔 |
| `Data/datasets/combined/validation_report.json` | 詳細驗證報告（JSON） |

## 故障排排除

### Q: 為什麼 feed 類有那麼多註解？
A: 因為 feed.yolo26 從 COCO 格式轉換時，一張圖片有多個 feed 檢測框（平均 ~13 個）。

### Q: 類別不平衡會影響訓練嗎？
A: 會的。使用 `auto_weight=True` 或 `class_weights` 參數來自動平衡損失函數。

### Q: 可以使用預訓練模型嗎？
A: 可以。使用 YOLO26s 預訓練模型會大大加快訓練和提高精度。

---

**最後更新**：2026-05-23  
**數據集版本**：1.0
