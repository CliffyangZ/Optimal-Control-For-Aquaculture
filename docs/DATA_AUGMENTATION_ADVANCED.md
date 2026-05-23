# 高級數據增強配置指南

## 📊 當前配置的增強

| 參數 | 值 | 描述 |
|-----|-----|------|
| `hsv_h` | 0.015 | 色調變化 (±1.5%) - **有輕微光線** |
| `hsv_s` | 0.7 | 飽和度變化 (±70%) - **有光線** |
| `hsv_v` | 0.4 | 亮度變化 (±40%) - **有光線** |
| `degrees` | 10.0 | 旋轉 (±10°) |
| `translate` | 0.1 | 平移 (±10%) |
| `scale` | 0.5 | 縮放 (±50%) |
| `flipud` | 0.1 | 垂直翻轉 (10%) |
| `fliplr` | 0.5 | 水平翻轉 (50%) |
| `mosaic` | 1.0 | 馬賽克增強 (100%) |
| `mixup` | 0.2 | 混合增強 (20%) |

## ⚠️ 缺失的增強

❌ **沒有明確的模糊處理**（Motion Blur, Gaussian Blur）  
❌ **沒有高級光線模擬**（Shadow, Exposure, Contrast）  
❌ **沒有噪聲**（Gaussian Noise, Salt & Pepper）

## ✅ 解決方案

### 方案 A：使用 YOLO 官方支持的參數（簡單）

YOLO 官方支持的完整增強參數列表：

```yaml
model:
  # 色彩空間增強
  hsv_h: 0.015      # ✓ 色調
  hsv_s: 0.7        # ✓ 飽和度 
  hsv_v: 0.4        # ✓ 亮度 (間接光線)
  
  # 空間變換
  degrees: 10.0     # ✓ 旋轉
  translate: 0.1    # ✓ 平移
  scale: 0.5        # ✓ 縮放
  perspective: 0.0  # ✓ 透視變換 (新增)
  flipud: 0.1       # ✓ 垂直翻轉
  fliplr: 0.5       # ✓ 水平翻轉
  
  # 高級增強
  mosaic: 1.0       # ✓ 馬賽克
  mixup: 0.2        # ✓ 混合
  copy_paste: 0.0   # ✓ 複製粘貼增強 (新增)
  erasing: 0.0      # ✓ 隨機擦除 (新增，類似模糊)
  crop_fraction: 1.0 # ✓ 隨機裁剪
```

### 方案 B：使用 Albumentations 自定義增強（推薦）

創建自訂增強 pipeline，包括模糊和光線：

```python
# custom_augmentation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_training_augmentation(img_size=832):
    """完整的訓練增強 pipeline"""
    return A.Compose([
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 光線增強
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        A.RandomBrightnessContrast(
            brightness_limit=0.3,    # ±30% 亮度
            contrast_limit=0.3,      # ±30% 對比度
            p=0.5
        ),
        A.RandomGamma(
            gamma_limit=(70, 130),   # 伽馬值範圍
            p=0.3
        ),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1), # 整個圖片
            p=0.3
        ),
        A.RandomFog(p=0.2),         # 霧霾效果
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 模糊增強
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        A.GaussianBlur(
            blur_limit=(3, 7),       # 模糊核心大小
            p=0.3
        ),
        A.MotionBlur(
            blur_limit=7,
            p=0.2
        ),
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 噪聲增強
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        A.GaussNoise(
            var_limit=(10.0, 50.0),  # 噪聲強度
            p=0.3
        ),
        A.ISONoise(p=0.2),           # ISO 噪聲
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 顏色和色調
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 空間變換
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        A.Rotate(
            limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.Perspective(
            scale=(0.05, 0.1),
            p=0.3
        ),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            p=0.5
        ),
        A.Flip(p=0.5),
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Resize
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        A.Resize(img_size, img_size, always_apply=True),
        A.Normalize(always_apply=True),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def get_val_augmentation(img_size=832):
    """驗證增強（最小化）"""
    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.Normalize(always_apply=True),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

### 方案 C：更新 YAML 配置（簡單提升）

編輯 `config/training.yaml`：

```yaml
model:
  # 原有參數...
  
  # 新增增強參數
  perspective: 0.001    # 透視變換
  erasing: 0.1          # 隨機擦除（類似小區域模糊效果）
  copy_paste: 0.0       # 複製粘貼增強
  
  # 加強光線增強
  hsv_v: 0.5            # 增加亮度變化（原為 0.4）
  hsv_s: 0.8            # 增加飽和度變化（原為 0.7）
```

## 📈 增強效果對比

### 當前配置效果
- ✓ 基本光線變化（HSV）
- ✓ 空間變換
- ✓ 馬賽克和混合
- ✗ 無模糊處理
- ✗ 無模擬真實場景噪聲

### 使用 Albumentations 後
- ✓ 完整光線模擬（亮度、對比度、陰影、霧）
- ✓ 多種模糊（高斯、運動）
- ✓ 真實噪聲（高斯、ISO）
- ✓ 色彩增強
- ✓ 所有空間變換

## 🎯 推薦配置

### 對於你的數據集（feed detection）

考慮到你的數據來自**水族館場景**，建議：

```yaml
# 強光線增強（水下光線變化）
hsv_h: 0.020      # 色調變化
hsv_s: 0.8        # 飽和度（水的顏色變化）
hsv_v: 0.5        # 亮度（水底光線變化）

# 強空間變換（不同角度）
degrees: 15
translate: 0.15
scale: 0.6
perspective: 0.001

# 增強增強
mosaic: 1.0
mixup: 0.2
copy_paste: 0.0
erasing: 0.1      # 加強
```

配合 Albumentations：
- 模糊：模擬水流造成的圖片模糊
- 噪聲：模擬水下相機噪聲
- 光線：模擬水下光線變化

## 🔧 實施步驟

1. **安裝 Albumentations**
   ```bash
   pip install albumentations
   ```

2. **創建自訂數據加載器**
   ```python
   from custom_augmentation import get_training_augmentation
   
   # 在你的訓練腳本中使用
   transform = get_training_augmentation(img_size=832)
   ```

3. **或更新 YAML 配置**
   ```bash
   # 編輯 config/training.yaml
   # 加入新增參數
   ```

## 📚 參考

- [Albumentations 文檔](https://albumentations.ai/)
- [YOLO 增強文檔](https://docs.ultralytics.com/tasks/detect/#train)
- [水下成像增強論文](https://arxiv.org/search/?query=underwater+image+enhancement)

---

**建議**：對於水族館 fish feed 檢測，**方案 B（Albumentations）最佳**，可以最好地模擬水下場景的光線和噪聲變化。
