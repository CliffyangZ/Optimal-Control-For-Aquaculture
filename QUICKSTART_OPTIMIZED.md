# 🚀 快速啟動優化訓練

## 準備工作

✅ 已完成：
- ✓ 數據合併（feed.yolo26 + FishTracker_v2）
- ✓ 配置優化（強光線增強、類別權重）
- ✓ 腳本創建（train_optimized.py）

## 立即開始訓練

### 方法 1：最簡單（推薦）

```bash
# 一行命令開始優化訓練
python scripts/train_optimized.py
```

**會發生什麼：**
1. 自動計算類別權重（處理 feed 占 83.5% 的不平衡）
2. 加載優化配置（強增強、自動混合精度）
3. 開始 300 epoch 的訓練
4. 每 10 epoch 保存一次模型
5. 自動選擇最佳模型

### 方法 2：使用 YOLO CLI

```bash
yolo detect train \
    data=data/datasets/combined/data.yaml \
    model=yolo26s.pt \
    epochs=300 \
    imgsz=832 \
    batch=16 \
    device=0 \
    auto_weight=True
```

### 方法 3：Python 代碼

```python
from ultralytics import YOLO

model = YOLO('yolo26s.pt')
results = model.train(
    data='data/datasets/combined/data.yaml',
    epochs=300,
    imgsz=832,
    batch=16,
    auto_weight=True,
)
```

## 期間發生了什麼優化

### 🎨 數據增強

|      | 原始 | 優化 | 效果 |
|------|------|------|------|
| 亮度 | 40% | **50%** ↑ | 更強光線變化 |
| 飽和度 | 70% | **80%** ↑ | 更強色彩變化 |
| 旋轉 | 10° | **15°** ↑ | 更多角度 |
| 新增 | - | **透視、擦除** | 真實場景模擬 |

### ⚖️ 類別平衡

```
自動計算的類別權重：
  feed       → 0.234 (最低，因為最多)
  fish       → 1.405
  debris     → 25.7
  lumpfish   → 286 (最高，因為最少)

結果：小類別不會被忽視 ✓
```

### 🔥 訓練優化

- **auto_weight**: True - 自動平衡損失函數
- **AMP**: True - 混合精度加速（快 30%）
- **Warmup**: 3 epoch - 穩定訓練
- **正則化**: weight_decay=0.0005 - 防止過擬合

## 訓練時間預期

```
Hardware: GPU (推薦 RTX 3090 或更好)
  - 1 epoch:  ~45-60 秒
  - 300 epochs: ~4-5 小時

硬件: CPU
  - 不推薦（>48 小時）
```

## 訓練期間監控

### 查看實時日誌

```bash
# 在訓練進行時，開新終端
tail -f runs/train/aquaculture_optimized/results.csv
```

### Weights & Biases 監控（如果已連接）

- 自動上傳到 wandb.ai
- 實時查看損失曲線、mAP 等

### TensorBoard 監控

```bash
tensorboard --logdir=runs/detect
# 在瀏覽器打開 http://localhost:6006
```

## 訓練後

### 1. 查看結果

```bash
# 最佳模型路徑
runs/train/aquaculture_optimized/weights/best.pt

# 結果文件
runs/train/aquaculture_optimized/results.csv
```

### 2. 評估模型

```python
from ultralytics import YOLO

model = YOLO('runs/train/aquaculture_optimized/weights/best.pt')
metrics = model.val(data='data/datasets/combined/data.yaml')
print(f"mAP@50: {metrics.box.map50:.3f}")
```

### 3. 使用模型推理

```bash
# 推理單張圖片
python train.py infer \
    --model=runs/train/aquaculture_optimized/weights/best.pt \
    --source=image.jpg

# 推理視頻
python train.py infer \
    --model=runs/train/aquaculture_optimized/weights/best.pt \
    --source=video.mp4 \
    --output=output.mp4
```

## 如果訓練中斷

### 恢復訓練

```bash
# 從最後保存的模型恢復（自動）
python scripts/train_optimized.py  # 會自動找到最後的 checkpoint

# 或手動指定
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/aquaculture_optimized/weights/last.pt')
model.train(resume=True)
"
```

## 預期結果

訓練完成後應該看到：

```
結果概覽：
┌─────────────┬────────────┬──────────┐
│ Metric      │ 預期值     │ 範圍     │
├─────────────┼────────────┼──────────┤
│ mAP@50      │ 0.78-0.82  │ 0.75-0.85│
│ Feed 精度   │ 0.85-0.90  │ 0.80-0.95│
│ Fish 精度   │ 0.70-0.80  │ 0.60-0.85│
│ Debris 精度 │ 0.50-0.65  │ 0.40-0.75│
└─────────────┴────────────┴──────────┘
```

## 故障排除

### Q: GPU 記憶體不足？
A: 減少 batch size：
```bash
python scripts/train_optimized.py --batch=8  # 從 16 改為 8
```

### Q: 訓練太慢？
A: 檢查是否使用 GPU：
```python
import torch
print(torch.cuda.is_available())  # 應該是 True
print(torch.cuda.get_device_name(0))  # 查看 GPU 名稱
```

### Q: 過擬合？
A: 增加增強強度：
```yaml
erasing: 0.2        # 從 0.1 增加
mixup: 0.3          # 從 0.2 增加
```

### Q: 欠擬合？
A: 減少正則化：
```yaml
weight_decay: 0.0001  # 從 0.0005 減少
```

## 完整工作流

```bash
# 1. 合併數據集（已完成）
python scripts/merge_datasets.py

# 2. 驗證數據（已完成）
python scripts/preprocess_dataset.py

# 3. 訓練模型（現在做）
python scripts/train_optimized.py

# 4. 評估結果
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/aquaculture_optimized/weights/best.pt')
model.val(data='data/datasets/combined/data.yaml')
"

# 5. 推理
python train.py infer \
    --model=runs/train/aquaculture_optimized/weights/best.pt \
    --source=test_image.jpg
```

## 常用命令速查

```bash
# 訓練
python scripts/train_optimized.py

# 評估
python -c "from ultralytics import YOLO; YOLO('runs/train/aquaculture_optimized/weights/best.pt').val(data='data/datasets/combined/data.yaml')"

# 推理
python train.py infer --model=runs/train/aquaculture_optimized/weights/best.pt --source=image.jpg

# 導出為 ONNX（邊緣計算）
python -c "from ultralytics import YOLO; YOLO('runs/train/aquaculture_optimized/weights/best.pt').export(format='onnx')"
```

---

**準備好了嗎？** 🎉

```bash
python scripts/train_optimized.py
```

開始訓練！
