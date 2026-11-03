# 📚 Kalman Filter Resources - 完整索引

## 文件清單

### 1. 📖 詳細技術文檔
**檔案:** `KALMAN_FILTER_EXPLANATION.md`

**內容:**
- ✅ 完整的數學推導
- ✅ 問題陳述和解決方案
- ✅ 系統模型詳解
- ✅ 算法的逐步說明
- ✅ 參數調整指南
- ✅ 實際例子 (step-by-step walkthrough)
- ✅ 應用案例：水產養殖飼料監控
- ✅ 參考文獻

**適用人群:** 需要深入理解算法的研究人員、講師

---

### 2. 🎨 海報版本（推薦用於展示）
**檔案:** `KALMAN_FILTER_POSTER.md`

**內容:**
- 🎯 視覺化的問題描述
- 📐 數學框架（簡化版）
- 🔄 演算法流程圖
- 📊 計算流程圖表
- 🔧 參數調整表格
- 📈 真實世界例子
- ✅ 驗證檢查清單
- 📋 演算法總結表

**適用人群:** 海報設計、學術報告、學生演講

---

### 3. 📋 快速參考卡
**檔案:** `KALMAN_FILTER_VISUAL_SUMMARY.txt`

**內容:**
- 🎯 問題為何需要 Kalman Filter
- 📐 狀態向量和轉移矩陣
- 🔄 5 步算法流程
- 🎯 核心方程式（適合貼海報）
- ⚙️ 參數選擇指南（表格形式）
- 📊 完整的例子演進過程
- ✅ 成功原因分析
- 📋 快速參考卡（ASCII 藝術）

**適用人群:** 快速查詢、海報設計、教學展示

---

## 使用建議

### 📌 如果你想...

#### 準備學術報告或論文
1. 先讀 `KALMAN_FILTER_EXPLANATION.md` 的「Problem Statement」和「Mathematical Framework」
2. 用 `KALMAN_FILTER_POSTER.md` 的圖表和表格
3. 在参考文献部分引用原始論文

#### 設計海報（PowerPoint / Poster）
1. 用 `KALMAN_FILTER_VISUAL_SUMMARY.txt` 的流程圖
2. 複製 `KALMAN_FILTER_POSTER.md` 的「Algorithm: Two-Phase Update Cycle」部分
3. 加入真實的飼料面積數據圖表（帶置信區間）
4. 參考「Parameter Tuning Guide」的表格

#### 教學 (課堂/研討會)
1. **5分鐘快速版:**
   - 用 `KALMAN_FILTER_POSTER.md` 的 「Problem & Solution」
   - 展示流程圖
   - 一個實例

2. **30分鐘詳細版:**
   - 整個 `KALMAN_FILTER_EXPLANATION.md`
   - 现场推导例子
   - 討論參數選擇

3. **60分鐘完整版:**
   - 包含上述所有內容
   - 代碼演示（用 Python Notebook）
   - 實驗結果分析
   - Q&A

#### 理解算法實現（程式設計）
1. 讀 `KALMAN_FILTER_EXPLANATION.md` 的「Algorithm: Two-Phase Update」
2. 對照 `analyze_feed_amount_rfdetr.ipynb` 中 `LogNormalKalmanFilter` 類的實現
3. 理解 `update()` 方法的各個步驟

---

## 關鍵公式速查

### 核心方程式（適合貼海報）

```
╔═══════════════════════════════════════════════════╗
║         PREDICT PHASE (預測)                     ║
╠═══════════════════════════════════════════════════╣
║  x̂⁻ = F · x̂                                    ║
║  P⁻ = F · P · F^T + Q                           ║
╚═══════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════╗
║         UPDATE PHASE (更新)                      ║
╠═══════════════════════════════════════════════════╣
║  γ = y - h · x̂⁻    (Innovation)                ║
║  S = h·P⁻·h^T + R   (Innovation covariance)    ║
║  K = (P⁻·h^T) / S   (Kalman gain)              ║
║  x̂ = x̂⁻ + K·γ     (State update)              ║
║  P = (I - K·h)·P⁻   (Covariance update)        ║
╚═══════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════╗
║         OUTPUT (輸出)                            ║
╠═══════════════════════════════════════════════════╣
║  Â = exp(μ̂ + P₁₁/2)    (Unbiased area)        ║
║  Rate = μ̇ × 100        (%/frame)              ║
║  CI = exp(μ̂ ± √P₁₁)    (Confidence interval)  ║
╚═══════════════════════════════════════════════════╝
```

---

## 參數快速選擇

### σ_R (Measurement Noise)

| Detector Accuracy | σ_R Value | Recommendation |
|-------------------|-----------|---|
| ±5%  | 0.049 | Too optimistic |
| ±10% | 0.095 | Good for high-end |
| ±15% | 0.140 | **Standard choice** ✓ |
| ±20% | 0.182 | Conservative |
| ±30% | 0.263 | Very skeptical |

**Default: σ_R = 0.15**

### σ_Q (Process Noise)

| Scenario | σ_Q Value | Behavior |
|----------|-----------|----------|
| Very smooth | 10⁻⁴ | Constant rate |
| Normal (Best) | **10⁻³** | Smooth with small changes |
| Variable | 5×10⁻³ | Medium sensitivity |
| Rapid changes | 10⁻² | Fast tracking |
| Too sensitive | 10⁻¹ | Follows noise |

**Default: σ_Q = 0.001**

---

## 實施檢查清單

在實施或使用 Kalman Filter 時，確認：

- [ ] 理解問題為什麼需要乘法噪聲模型
- [ ] 選擇適當的 σ_R（基於檢測器精度）
- [ ] 選擇適當的 σ_Q（基於預期的動態變化）
- [ ] 第一幀用測量值初始化狀態向量
- [ ] 實現 5 步算法：Predict → Innovate → Gain → Update → Output
- [ ] 應用 Log-Normal 偏差校正 (+P₁₁/2)
- [ ] 計算並展示置信區間
- [ ] 監控創新值（γ）以檢測異常
- [ ] 驗證濾波結果的光滑性
- [ ] 評估與基線的改進

---

## 推薦學習路徑

### 🟢 初級（想快速理解）
1. 讀 `KALMAN_FILTER_VISUAL_SUMMARY.txt`（10分鐘）
2. 看 `analyze_feed_amount_rfdetr.ipynb` 運行結果（5分鐘）
3. 看 Jupyter notebook 的可視化圖表（5分鐘）

**總耗時:** ~20 分鐘

---

### 🟡 中級（想深入理解）
1. 讀 `KALMAN_FILTER_POSTER.md`（20分鐘）
2. 研究 `KALMAN_FILTER_EXPLANATION.md` 的 Algorithm 部分（30分鐘）
3. 在 Jupyter notebook 中逐步執行程式碼，理解每個步驟（30分鐘）

**總耗時:** ~80 分鐘

---

### 🔴 進階（想完全掌握）
1. 深入閱讀 `KALMAN_FILTER_EXPLANATION.md`（60分鐘）
2. 推導 Example 中的計算（30分鐘）
3. 修改程式碼改變參數，觀察效果（30分鐘）
4. 閱讀原始論文參考（可選，60分鐘）

**總耗時:** ~180 分鐘

---

## 常見問題解答

### Q1: 為什麼需要 log transform？
**A:** 檢測器的誤差是乘法性的（相對誤差），不是加法性的。
- 原始: 50% of mean ± relative_error
- Log: additive Gaussian noise

### Q2: 為什麼要加 P₁₁/2？
**A:** 這是 Log-Normal 分佈的偏差校正。
- 沒有: 估計會系統性低估 5-10%
- 有: E[A] = exp(μ + σ²/2) 是無偏的

### Q3: Kalman Gain 的意義？
**A:** K ∈ [0,1]，決定信任測量的程度
- K → 1: 新測量可信，調整量大
- K → 0: 預測更可信，調整量小

### Q4: 如何選擇 σ_Q？
**A:** 根據消耗速率的變化速度
- 恆定: σ_Q = 10⁻⁴
- 緩慢變化: σ_Q = 10⁻³ ✓ (推薦)
- 快速變化: σ_Q = 10⁻²

### Q5: 可以用於其他應用嗎？
**A:** 是！任何乘法噪聲的物理量都適用
- 水溫變化
- 光照強度
- 有機物濃度

---

## 相關代碼檔案

### Python Notebooks
- **`analyze_feed_amount_rfdetr.ipynb`** - RF-DETR + Kalman Filter 完整實現
- **`analyze_feed_amount.ipynb`** - YOLO + Kalman Filter 版本

### 推理模組
- **`core/infer/rfdetr_detector.py`** - RF-DETR 推理包裝
- **`core/infer/detector.py`** - YOLO 推理包裝

### 配置檔案
- **`src/config/training.yaml`** - 訓練配置
- **`pyproject.toml`** - 專案依賴

---

## 如何在海報上使用

### 推薦版面配置

```
┌─────────────────────────────────────────────────────┐
│          TITLE: Log-Normal Kalman Filter            │
│     for Aquaculture Feed Monitoring                 │
├─────────────────────────────────┬───────────────────┤
│  Problem & Solution (left side)  │  Flow Diagram     │
│  ✓ Multiplicative noise          │  (right side)     │
│  ✓ Log transform solution        │  PREDICT ↓        │
│  ✓ Why Kalman Filter            │  INNOVATE ↓       │
│                                  │  GAIN ↓           │
├─────────────────────────────────┼───────────────────┤
│  Core Equations (center)         │  Real Results     │
│  [Print from VISUAL_SUMMARY.txt] │  (plots/images)   │
│                                  │                   │
├─────────────────────────────────┴───────────────────┤
│  Parameter Selection Table (footer)                 │
│  σ_R recommendations | σ_Q recommendations          │
└─────────────────────────────────────────────────────┘
```

### 顏色建議
- **Blue**: PREDICT phase
- **Green**: UPDATE phase  
- **Red**: OUTPUT
- **Gray**: Background/annotations

### 字體建議
- 標題: 36pt bold
- 小標題: 24pt bold
- 方程式: 18pt monospace
- 註解: 14pt regular

---

## 版本歷史

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-23 | 1.0 | Initial creation |
| - | 1.1 | Added visual summary |
| - | 1.2 | Added poster version |

---

## 許可證和引用

如果在論文、報告或展示中使用，請引用：

```
@misc{aquaculture_kalman_2026,
  title={Log-Normal Kalman Filter for Feed Monitoring},
  author={Your Name},
  year={2026},
  organization={Your Organization}
}
```

---

**Last Updated:** 2026-05-23
**Location:** `/home/xinze/Program/Aquaculture_CV_Project/`

