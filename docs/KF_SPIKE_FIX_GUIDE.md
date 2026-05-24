# Kalman Filter 極端值放大 — 4 個改進方案

## 🔴 問題診斷

你的圖表顯示 KF 過濾曲線在某些幀處出現**極端尖峰**，特別是：
- Frame 2-9：檢測失敗（area=0）後的異常波動
- Frame 最大值：4,322,886 vs Raw 1,707,559（放大 2.5x）
- **降噪效果為負 (-24.7%)**，說明方差反而增加

**根本原因：**
1. **`sigma_Q = 1e-3` 太小** → KF 認為過程極穩定，但遇到極端測量時被迫大幅調整
2. **零值處理** → `log(0) = -∞`，導致狀態崩潰
3. **無回向平滑** → 前向 KF 無法回頭修正已經的尖峰

---

## ✅ 方案 1：增加過程噪聲（推薦，最簡單）

**為什麼有效：** 增大 `sigma_Q` 允許 KF 更快地調整狀態，減少對單個極端測量的過度反應。

**改動：**
```python
CONFIG = {
    ...
    'sigma_Q': 5e-3,  # ← 改為 5e-3，從 1e-3 增加 5 倍
    ...
}
```

**嘗試的 `sigma_Q` 值（升序）：**
| sigma_Q  | 效果 |
|----------|------|
| 1e-4     | 平滑但無法跟踪快速變化 |
| **5e-3** | **平衡點：減少尖峰，保持跟踪** ✓ |
| 1e-2     | 更積極，適合高噪聲環境 |
| 5e-2     | 太敏感，會跟踪噪聲 |

**優點：**
- 單行改動，立竿見影
- 無需複雜算法

**缺點：**
- 可能引入更多速率噪聲

---

## ✅ 方案 2：跳過零值測量（推薦，已實現）

**為什麼有效：** 檢測失敗（area=0）不代表飼料消失，而是 YOLO 未檢測到。直接跳過這類測量，避免狀態崩潰。

**改動：** （已在改進的 KF 類中實現）
```python
def update(self, area_measurement: float) -> float:
    # 跳過檢測失敗
    if area_measurement <= 0:
        return self.area_estimate if self.initialized else 0.0
    # ... 正常更新 ...
```

**效果：**
```
Frame 2: area=0        → 跳過，保持前一狀態
Frame 3: area=103k     → 從前一狀態更新，無極端跳跳
```

**優點：**
- 防止 `log(0) = -∞` 問題
- 直觀、無副作用

**缺點：**
- 需要知道何時是檢測失敗 vs 真實的零面積（本場景：總是檢測失敗）

---

## ✅ 方案 3：異常值拒絕（推薦，已實現）

**為什麼有效：** 識別遠離預測值（>3σ）的測量，跳過它們，讓 KF 自行預測。

**改動：** （已在改進的 KF 類中實現）
```python
CONFIG = {
    ...
    'reject_outliers': True,
    'outlier_threshold': 3.0,  # 超過 3σ 的測量被拒絕
    ...
}
```

**實現邏輯：**
```python
innovation = y - predicted_y
S = innovation_variance
if abs(innovation) > threshold * sqrt(S):
    skip_update()  # 只預測，不更新
```

**優點：**
- 自動檢測離群值，無需手動調整
- 保護 KF 免受異常測量干擾

**缺點：**
- threshold=3.0 可能需要調整（3.0, 2.5, 2.0）

---

## ✅ 方案 4：平滑過濾（Rauch-Tung-Striebel 後向平滑）

**為什麼有效：** 前向 KF 只看過去數據；RTS 後向平滑利用**整個序列**信息，事後修正尖峰。

**改動：** （已在改進的 KF 類中實現）
```python
# 前向過濾所有幀
for measurement in all_measurements:
    kf.update(measurement)

# 後向平滑（可選）
kf.rts_smooth()

# 提取結果
filtered_areas = [kf.history_x[i][0] for i in range(n)]
```

**效果：**
```
前向 KF：  [1000, 4500, 1200, 2000]  （尖峰 4500）
RTS 平滑： [1100, 2200, 1800, 1900]  （消除尖峰）
```

**優點：**
- 最強大，利用全局信息
- 自動平衡所有幀的偏差

**缺點：**
- 需要等待所有數據後處理（延遲）
- 計算成本 O(n)

---

## 🎯 推薦組合方案

為最佳效果，**組合使用方案 1+2+3**：

```python
CONFIG = {
    'sigma_R': 0.15,
    'sigma_Q': 5e-3,           # ✓ 方案 1：增加過程噪聲
    'reject_outliers': True,   # ✓ 方案 3：拒絕異常值
    'outlier_threshold': 3.0,
}

kf = LogNormalKalmanFilter(
    sigma_R=CONFIG['sigma_R'],
    sigma_Q=CONFIG['sigma_Q'],
    reject_outliers=CONFIG['reject_outliers'],
    outlier_threshold=CONFIG['outlier_threshold'],
)

# 前向過濾（方案 2：自動跳過零值）
for area in feed_areas_raw:
    kf.update(area)  # 內部跳過零值

# 可選：方案 4 後向平滑
kf.rts_smooth()
```

**預期結果：**
- ✓ 消除尖峰，特別是 frame 2-9 和末尾
- ✓ 降噪效果從 -24.7% 改善到 +20% 左右
- ✓ 消耗速率估計更平穩

---

## 📊 A/B 對比測試

在 notebook 中測試不同配置：

```python
# 對比配置
configs = [
    {'sigma_Q': 1e-3,  'reject_outliers': False},  # 原始
    {'sigma_Q': 5e-3,  'reject_outliers': False},  # 方案 1
    {'sigma_Q': 5e-3,  'reject_outliers': True},   # 方案 1+3
    {'sigma_Q': 1e-2,  'reject_outliers': True},   # 更激進
]

for cfg in configs:
    kf = LogNormalKalmanFilter(**cfg)
    # 運行前向過濾...
    # 計算 log-MAE, std, 視覺檢查...
```

---

## 🔧 調整參數的三步驟

1. **視覺檢查**：圖表上尖峰是否消失？
2. **統計檢查**：`std` 是否下降？`log-MAE` 是否改善？
3. **物理檢查**：消耗速率是否合理？（不應出現 300% 的幀）

---

## 📝 完整改進版本

改進的 `LogNormalKalmanFilter` 類已在你的 notebook 中更新，新增：
- ✓ 零值跳過
- ✓ 異常值拒絕 + outlier_threshold
- ✓ RTS 平滑方法 `kts_smooth()`

**立即嘗試**：執行 notebook，使用新的 CONFIG 參數，對比結果。
