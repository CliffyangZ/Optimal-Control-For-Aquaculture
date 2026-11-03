# 📊 Log-Normal Kalman Filter for Feed Monitoring
## 水產養殖飼料量監控系統

---

## 🎯 Problem & Solution

### ❌ Problem
- YOLO/RF-DETR 檢測結果有**乘法性噪聲**
- 誤差與目標大小成正比 (±10~20%)
- 無法直接用於精確監控飼料量

### ✅ Solution
使用 **Log-Normal Kalman Filter**：
1. 將乘法噪聲轉換為加法噪聲 (log transform)
2. 用 Kalman Filter 進行最優濾波
3. 聯合估計飼料面積 + 消耗速率
4. 提供置信區間和無偏估計

---

## 📐 Mathematical Model

### 1. State Vector (狀態向量)

```
        ┌─────────────────────────┐
        │  x = [ μ,  μ̇ ]ᵀ        │
        ├─────────────────────────┤
        │ μ  = log(真實飼料面積)   │
        │ μ̇  = 消耗速率 [1/幀]   │
        └─────────────────────────┘
```

**含義：**
- **μ** : 飼料面積的對數值（工作空間）
- **μ̇** : 每幀的消耗百分比
  - μ̇ = -0.02 → 每幀減少 2% (食物消失)
  - μ̇ = +0.01 → 每幀增加 1% (堆積)

### 2. System Dynamics (系統轉移方程)

```
┌────────────────────────────────────────────┐
│  x_{t+1} = F · x_t + w_t                  │
├────────────────────────────────────────────┤
│            ┌─────────┐                     │
│  F =       │ 1   Δt  │   (constant-velocity)
│            │ 0   1   │    Δt = 1 frame    │
│            └─────────┘                     │
│                                            │
│  w_t ~ N(0, Q)  (process noise)           │
│  Q = σ_Q · [ Δt⁴/4  Δt³/2 ]              │
│             [ Δt³/2  Δt²   ]              │
└────────────────────────────────────────────┘
```

**直觀解釋：**
```
下一時刻的面積 = 當前面積 + 消耗速率 × 時間
（在對數空間中）
```

### 3. Observation Model (觀測模型)

```
┌──────────────────────────────────────┐
│ 原始測量 (detector):                │
│     z_t  [pixel²]  (bounding box)   │
│                                      │
│ 對數空間轉換:                         │
│     y_t = log(z_t)                  │
│          = μ_t + v_t                │
│     v_t ~ N(0, σ_R²)                │
│            ↑ measurement noise       │
│                                      │
│ 觀測矩陣:                            │
│     h = [1, 0]  (只測量面積)        │
└──────────────────────────────────────┘
```

---

## 🔄 Algorithm: Two-Phase Update Cycle

### Phase 1️⃣: PREDICT (預測)

```
┌─────────────────────────────────────────┐
│ Before seeing new measurement:          │
│                                         │
│  x̂⁻_{t|t-1} = F · x̂_{t-1|t-1}         │
│                                         │
│  P⁻_{t|t-1}  = F · P_{t-1|t-1} · Fᵀ + Q│
│                ↑        ↑         ↑      │
│         prev state  covariance  uncertainty│
│                                         │
│  Result: 先驗估計 (prior estimate)     │
└─────────────────────────────────────────┘

x̂⁻  = predicted state
P⁻  = predicted covariance (uncertainty)
```

### Phase 2️⃣: UPDATE (更新)

```
┌──────────────────────────────────────────┐
│ After seeing new measurement:            │
│                                          │
│ 1. Innovation (測量驚訝度):               │
│    γ_t = y_t - h · x̂⁻_{t|t-1}          │
│         = log(z_t) - μ̂⁻_t              │
│                                          │
│ 2. Innovation Covariance:                │
│    S_t = h · P⁻_{t|t-1} · hᵀ + R       │
│        = σ²_預測 + σ²_測量              │
│         (total uncertainty)              │
│                                          │
│ 3. Kalman Gain (信任度):                 │
│    K_t = (P⁻_{t|t-1} · hᵀ) / S_t       │
│        ∈ [0, 1]                         │
│    • K → 1: 信任新測量                  │
│    • K → 0: 信任預測                    │
│                                          │
│ 4. State Update:                         │
│    x̂_{t|t} = x̂⁻_{t|t-1} + K_t · γ_t  │
│              └─ prior ┘  └ correction ┘│
│                                          │
│ 5. Covariance Update:                    │
│    P_{t|t} = (I - K_t · h) · P⁻_{t|t-1}│
│                                          │
│  Result: 後驗估計 (posterior estimate)  │
└──────────────────────────────────────────┘
```

---

## 📊 Computational Flow

```
┌─────────────────────────────────────────────────────┐
│              Frame t Measurement                    │
│              z_t (pixel²)                          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  y_t = log(z_t)      │  ◄── Log Transform
          │                      │
          └──────────┬───────────┘
                     │
                     ▼
      ┌──────────────────────────────┐
      │     PREDICT PHASE            │
      │  (使用上一幀的後驗估計)       │
      │                              │
      │  x̂⁻ = F · x̂                 │
      │  P⁻ = F·P·Fᵀ + Q             │
      └──────────┬───────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │      UPDATE PHASE                │
    │  (融合新的測量)                  │
    │                                  │
    │  γ = y - h·x̂⁻                   │
    │  S = h·P⁻·hᵀ + R                │
    │  K = (P⁻·hᵀ) / S                │
    │  x̂ = x̂⁻ + K·γ                  │
    │  P = (I - K·h)·P⁻                │
    └──────────┬────────────────────────┘
               │
               ▼
    ┌─────────────────────────────────┐
    │   OUTPUT (Convert to Pixels)    │
    │                                  │
    │  Ê[A_t] = exp(μ̂_t + P₁₁/2)    │
    │           └─ bias correction ─┘ │
    │                                  │
    │  Rate = μ̇_t × 100 (%/frame)    │
    │  Confidence = exp(μ̂ ± √P₁₁)    │
    └─────────────────────────────────┘
               │
               ▼
        ┌──────────────────┐
        │  Filtered Area   │
        │  Consumption     │
        │  Confidence Band │
        └──────────────────┘
```

---

## 🔧 Parameter Tuning Guide

### σ_R: Measurement Noise (測量噪聲)

**定義：** 檢測器的相對誤差

```
σ_R = ln(1 + relative_error)

┌──────────────────────────────────────┐
│ Detector Error  │  σ_R Value        │
├──────────────────────────────────────┤
│  ±5%           │  0.049            │
│  ±10%          │  0.095            │
│  ±15%          │  0.140  ◄─ typical│
│  ±20%          │  0.182            │
│  ±30%          │  0.263            │
└──────────────────────────────────────┘
```

**效果：**
```
Small σ_R (< 0.1)  →  高度信任檢測器 → 濾波結果嘈雜
Large σ_R (> 0.3)  →  低度信任檢測器 → 反應遲鈍

建議：σ_R = 0.10 ~ 0.20
```

### σ_Q: Process Noise (過程噪聲)

**定義：** 消耗速率的變化許可程度

```
σ_Q = Intensity of rate-of-change

┌──────────────────────────────────────┐
│ σ_Q Value   │  Characteristics      │
├──────────────────────────────────────┤
│ 10⁻⁴       │  速率恆定，高度光滑  │
│ 10⁻³       │  溫和變化，推薦值    │ ◄─
│ 10⁻²       │  快速變化，敏感      │
│ 10⁻¹       │  極快變化，跟蹤噪聲  │
└──────────────────────────────────────┘
```

**推薦設置：**
```
一般監控：σ_Q = 1.0 × 10⁻³
快速變化：σ_Q = 5.0 × 10⁻³  
光滑濾波：σ_Q = 1.0 × 10⁻⁴
```

---

## 📈 Example: Real-World Scenario

### 飼料消耗過程

```
Time    Raw Area    Filtered    Rate        Status
        (noisy)     (smooth)    (%/frame)
─────────────────────────────────────────────────────
0       500±50      485         0.00        initialization
1       480±45      492         -0.15       consuming
2       510±50      498         +0.10       slight accumulation
3       475±45      495         -0.08       back to consuming
...
100     300±40      310         -0.85       fast consumption
...
200     80±15       92          -1.20       very fast
...
300     5±2         5.5         ≈0          plateau (depleted)
```

**決策規則：**
```
IF filtered_area < 10 px² AND rate < -0.5 %/frame
THEN refill feed
```

---

## 🎓 Why Log-Normal Kalman Filter?

### 1. **Noise Model Accuracy** ✓
- 檢測噪聲確實是乘法性的
- Log 轉換使其變為加法性 → KF 最優

### 2. **Joint Estimation** ✓
- 面積和速率同時估計
- 避免有限差分的延遲和噪聲

### 3. **Unbiased Output** ✓
```
Standard: E[A] = E[e^μ] ≠ e^(E[μ])  (biased)
Corrected: Ê[A] = exp(μ̂ + P₁₁/2)   (unbiased)
                              ↑
                        bias correction
```

### 4. **Uncertainty Quantification** ✓
- 從協方差獲得置信區間
- 知道估計的可靠性

---

## 📋 Algorithm Summary Table

| Phase | Operation | Input | Output | Purpose |
|-------|-----------|-------|--------|---------|
| **Predict** | x̂⁻ = F·x̂<br>P⁻ = F·P·Fᵀ+Q | x̂, P (previous) | x̂⁻, P⁻ | 根據動態預測下一刻 |
| **Innovate** | γ = y - h·x̂⁻<br>S = h·P⁻·hᵀ + R | y (measurement)<br>x̂⁻, P⁻ | γ, S | 計算測量驚訝度 |
| **Gain** | K = (P⁻·hᵀ)/S | P⁻, h, S | K ∈ [0,1] | 決定相信程度 |
| **Update** | x̂ = x̂⁻ + K·γ<br>P = (I-K·h)·P⁻ | x̂⁻, P⁻, K, γ | x̂, P (posterior) | 融合測量與預測 |
| **Output** | Â = exp(μ̂ + P₁₁/2)<br>R̂ = μ̇ | x̂, P | Area, Rate | 轉換回像素空間 |

---

## 🔑 Key Equations Reference

```
╔════════════════════════════════════════════════════╗
║          CORE KALMAN FILTER EQUATIONS              ║
╠════════════════════════════════════════════════════╣
║                                                    ║
║  PREDICT:                                         ║
║  • x̂⁻ = F·x̂          (state prediction)          ║
║  • P⁻ = F·P·Fᵀ + Q    (covariance prediction)     ║
║                                                    ║
║  UPDATE:                                          ║
║  • γ = y - h·x̂⁻      (innovation/residual)       ║
║  • S = h·P⁻·hᵀ + R    (innovation covariance)    ║
║  • K = (P⁻·hᵀ)/S      (Kalman gain)              ║
║  • x̂ = x̂⁻ + K·γ      (state update)             ║
║  • P = (I - K·h)·P⁻   (covariance update)        ║
║                                                    ║
║  OUTPUT:                                          ║
║  • Ê[A] = exp(μ̂ + P₁₁/2)  (area estimate)       ║
║  • Conf = exp(μ̂ ± √P₁₁)    (95% interval)       ║
║  • Rate = μ̇ × 100    (%/frame)                  ║
║                                                    ║
╚════════════════════════════════════════════════════╝
```

---

## 📚 Literature & References

1. **Kalman (1960):** A new approach to linear filtering and prediction
2. **Bar-Shalom et al. (2001):** Estimation with Applications to Tracking
3. **Log-Normal Distribution:** Perfect for multiplicative noise scenarios
4. **Bias Correction:** Essential when working with exponential transforms

---

## ✅ Validation Checklist

- [ ] σ_R 根據檢測器精度設置 (0.10~0.20)
- [ ] σ_Q 根據動態變化選擇 (1e-3 推薦)
- [ ] 第一幀用測量值初始化
- [ ] 每次更新計算置信區間
- [ ] 輸出時應用偏差校正 (P₁₁/2)
- [ ] 監控創新γ_t 檢查異常值
- [ ] 週期性檢視濾波器性能

---

**For Poster Display:**
- Print with high contrast (dark background recommended)
- Highlight key formulas in color boxes
- Use flow diagrams for Algorithm section
- Include real example plots from your aquaculture data
