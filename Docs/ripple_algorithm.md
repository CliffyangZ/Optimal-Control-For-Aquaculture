# Spatiotemporal Ripple Detection Algorithm — v2

水產養殖監控系統的水面活動偵測演算法（工業級版本）。實作於 [core/ripple_algo.py](../src/core/ripple_algo.py)。

> **v1 → v2 主要差異**：
>
> - **Cell center → Candidate center**：不再假設網格中心是漣漪中心，先用動態證據找候選點，再在候選點周圍做徑向評分。
> - **Ring average → Radius-bin peak**：把 `[r_min, r_max]` 切成多個半徑 bin，取最強 bin 的 top-k 平均，避免局部小漣漪被整環稀釋。
> - **Linear sum → Gated fusion**：`raw_score = activity × structure`，沒有結構就不會因為像素閃爍累積成中等分數。
> - **Hard MOG2 mask → Soft MOG2 + 自適應學習率**：背景區只是降權不是歸零；活動高時凍結背景模型，避免持續水紋被吸收。
> - **Specular highlight mask**：HSV 範圍偵測水面反光，提早遮蔽。
> - **Flow divergence**：除了 outward dot product，再加上 `∂u/∂x + ∂v/∂y` 的正值區。
> - **Soft saturation**：`x / (x + cap)` 取代 `clip(x / cap, 0, 1)`，沒有 cap 後的平台期。

---

## 1. 目標

從水面攝影機影像中即時偵測**生物造成的漣漪活動**（魚的游動、餵食反應、突發行為），輸出每個網格區塊的活動分數與全幀熱力圖（heatmap），用於：

- 即時顯示養殖池中的活動熱區（fishing hotspot）
- 統計短期 / 中期 / 長期活動程度
- 觸發異常行為告警（過度活躍 / 異常靜止）

---

## 2. 系統架構（v2）

```
┌──────────────────────────────────────────┐
│           Input BGR Frame                 │
└──────────────────┬───────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Preprocess         │  ←  gray + blur + CLAHE
        │  Specular mask      │  ←  HSV 高亮抑制 → valid_mask
        │  Optional ROI mask  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ MOG2 (adaptive lr)  │  ←  活動高時 lr=0
        │ Soft fg mask        │  ←  bg 區降權至 bg_soft_floor
        └──────────┬──────────┘
                   │
   ┌───────────────┼───────────────────────────────┐
   ▼               ▼                               ▼
┌────────┐  ┌──────────────┐                ┌─────────────┐
│ Canny  │  │ Sobel gx, gy │                │ DIS Optflow │
│ edges  │  │              │                │ + divergence│
└───┬────┘  └──────┬───────┘                └──────┬──────┘
    │              │                               │
    └──────────────┴────────────┬──────────────────┘
                                │
            ┌───────────────────▼─────────────────────┐
            │      Dynamic Evidence Map                │
            │      (motion ∪ edge_xor) ∧ soft_fg≥0.5   │
            └───────────────────┬─────────────────────┘
                                │
                                ▼
                ┌───────────────────────────┐
                │  Connected Components      │
                │  → Candidate Centers (top N)│
                └─────────────┬─────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ Radius-bin   │    │ Radial flow      │    │ Divergence    │
│ peak edge    │    │ outward score    │    │ ring pool     │
│ alignment    │    │                  │    │ (soft sat)    │
└──────┬───────┘    └────────┬─────────┘    └──────┬───────┘
       │                     │                     │
       └─────────────┬───────┴──────────────┬──────┘
                     │                      │
            spatial_peak             expansion = w_rf·radial_flow
                     │                       + w_div·div_norm
                     ▼                      ▼
            ┌────────────────────────────────────┐
            │  structure = 0.6·spatial + 0.4·exp │  ← linear (default)
            │            or sqrt(spat·exp)       │  ← geomean (strict)
            └─────────────────┬──────────────────┘
                              │
                              ▼
              ┌─────────────────────────────┐
              │ Render Gaussian blob heatmap│  ← amplitude = structure
              └────────────────┬────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ structure_grid = max-  │
                  │ pool over R×C cells    │
                  └─────────────┬──────────┘
                                │
   ┌────────────────────────────┼────────────────────────────┐
   │                                                         │
   ▼                                                         ▼
┌────────────────────┐                          ┌──────────────────────┐
│ motion_density (·) │   ──── soft_saturate ──> │ activity_grid (R×C)  │
│ edge_density   (·) │                          │ (motion + edge mix)  │
└────────────────────┘                          └──────────┬───────────┘
                                                           │
                                                           ▼
                                         ┌─────────────────────────────────┐
                                         │ raw_score = activity·structure  │
                                         └─────────────────┬───────────────┘
                                                           │
                                                           ▼
                                          ┌──────────────────────────────┐
                                          │ Sliding windows              │
                                          │ short / mid / long  averages │
                                          └──────────────┬───────────────┘
                                                         │
                                                         ▼
                                              ┌────────────────────┐
                                              │ ActivityResult     │
                                              │ heatmap + scores   │
                                              └────────────────────┘
```

核心理念：

> **漣漪 = 「徑向結構（同心圓邊緣 + 向外光流 + 正散度）」+「夠強的局部活動」**
>
> 兩者必須**同時且共處**才會得到高分。線性相加會讓單純的像素閃爍、相機抖動、反光累積成中等分數；v2 用乘法門控避免這個漏洞。

---

## 3. 處理流程

### 3.1 Preprocess

`_preprocess()`：BGR → 灰階 → Gaussian blur (`blur_kernel`) → CLAHE (`clahe_clip`/`clahe_tile`)。

### 3.2 Specular Highlight Mask

`_specular_mask()`：在 HSV 空間中找「亮度高、飽和度低」的水面反光區域：

```
spec = (V ≥ v_thresh) ∧ (S ≤ s_thresh)
```

膨脹 `specular_dilate` px 後得到 `valid_mask = 1 - spec`。後續所有訊號都會乘上 `valid_mask`。

### 3.3 MOG2 + Soft Foreground

```
fg_raw  = MOG2.apply(frame, learningRate=lr)        # lr 自適應
soft_fg = bg_soft_floor + (1 − bg_soft_floor)·(fg_raw / 255)
soft_fg = soft_fg × valid_mask
```

- **自適應學習率**：上一幀的 `mean(activity_grid)` 若超過 `bg_freeze_activity`（預設 0.30），本幀就把 `learningRate=0`，避免魚群活躍時被 MOG2 學進背景。
- **Soft mask（不是 hard AND）**：背景區仍保留 `bg_soft_floor`（預設 0.30）的權重，讓持續水紋訊號不會完全歸零。

### 3.4 邊緣與梯度

- `_canny_edges()`：整張只算一次。
- `_sobel_gradients()`：產出 `gx`, `gy`，提供徑向對齊度計算。

### 3.5 光流與散度

- `_optical_flow()`：DIS PRESET_FAST，在 `flow_scale` 縮圖上算，再 resize 回原尺寸並把向量值還原。
- `_compute_flow_divergence()`：對 `flow` 取 `∂u/∂x + ∂v/∂y` 後 clip 到正值。擴張源附近散度為正，旋轉/平移運動則不一定。

### 3.6 動態證據與候選點

```
motion_mask = |gray − prev_gray| > diff_threshold
edge_xor    = canny_t XOR canny_{t-1}
active_pix  = (motion_mask ∨ edge_xor) ∧ (soft_fg ≥ 0.5)
```

對 `active_pix` 做 morphological close → `connectedComponentsWithStats` → 取面積 ≥ `candidate_min_area` 的前 `candidate_max_count` 個 component，centroid 即候選漣漪中心。

> **解掉「cell 中心不等於漣漪中心」的根本問題。**

### 3.7 每個候選點的結構分數

對每個候選點 `(cx, cy)`：

#### 3.7.1 Spatial（Radius-Bin Peak）

把 `[r_min, r_max]` 切成 `radius_bins` 個等寬半徑帶。每個帶內：

```
alignment = |r̂ · g| / |g|   (僅 edge 像素)
bin_score = mean(alignment)
```

最終 `spatial = mean(top-bin_topk of bin_scores)`。

> 小漣漪只佔幾個半徑帶 → 全環平均會稀釋訊號；分 bin 取最強就不會。

#### 3.7.2 Expansion（Radial Flow + Divergence）

```
radial_flow  = _radial_expansion_score(flow, cx, cy)
                                  # outward 加權平均，weight = clip(|flow|/2, 0, 1)
div_ring     = _ring_pool_mean(div_pos, cx, cy)
div_norm     = soft_saturate(div_ring, divergence_cap)
expansion    = (w_radial_flow·radial_flow + w_divergence·div_norm) / (w_rf + w_div)
```

#### 3.7.3 Structure（兩種模式）

```
linear  (default): structure = (w_spatial·spatial + w_expansion·expansion) / Σw
geomean (strict) : structure = sqrt(spatial · expansion)
```

`structure_combine="linear"` 預設較寬鬆；`"geomean"` 要求徑向結構與向外擴張**同時**存在，誤判率低但對輕微擾動較不敏感。

### 3.8 候選點 → Heatmap → Grid 池化

對每個候選點以 `candidate_sigma` 為標準差畫一個 2D Gaussian blob（amplitude = `structure`），多個候選點取 pixel-wise max：

```
structure_hm  = max_blend(Gaussian(cx, cy, σ=candidate_sigma) × structure)
structure_grid[r, c] = max(structure_hm in cell (r, c))
```

### 3.9 Per-Cell Activity（Soft Saturation）

```
motion_field = motion_mask × soft_fg
edge_field   = edge_xor   × soft_fg
motion_density = mean(motion_field) per cell
edge_density   = mean(edge_field)   per cell
motion_soft = motion_density / (motion_density + motion_cap)
edge_soft   = edge_density   / (edge_density   + edge_cap)
activity_grid = (w_motion·motion_soft + w_edge·edge_soft) / (w_motion + w_edge)
```

> Soft saturation（`x / (x + cap)`）取代 v1 的 hard clip，避免 cap 後分不出強度差異。

### 3.10 Gated Fusion

```
raw_score = activity_grid × structure_grid    ∈ [0, 1]
```

- 有動態無結構（像素閃爍、反光抖動）→ structure ≈ 0 → raw_score ≈ 0
- 有結構無動態（靜態反光紋路）→ activity ≈ 0 → raw_score ≈ 0
- 兩者都中等：activity=0.4, structure=0.4 → raw=0.16（medium 邊界）
- 兩者都強：activity=0.7, structure=0.6 → raw=0.42（high 邊界）

### 3.11 時序聚合

不變：每個 cell 維護 short / mid / long 三個 deque：

| 窗口 | 預設長度 | 物理意義 |
|---|---|---|
| `short_window` | 30 幀（≈1 秒） | 立即性擾動 |
| `mid_window` | 150 幀（≈5 秒） | 短期活動 |
| `long_window` | 900 幀（≈30 秒） | 穩定 hotspot |

---

## 4. 輸出

### `ActivityResult` dataclass

| 欄位 | 型別 | 說明 |
|---|---|---|
| `frame_index` | int | 累計處理幀數 |
| `raw_score` | `(R, C)` float32 | 當幀原始分數 |
| `short_score` | `(R, C)` float32 | 1 秒平均 |
| `mid_score` | `(R, C)` float32 | 5 秒平均 |
| `long_score` | `(R, C)` float32 | 30 秒平均 |
| `heatmap` | `(H, W)` float32 ∈ [0,1] | mid_score 雙線性升取樣 + 候選點 Gaussian 強化 |
| `candidates` | `list[(cx, cy, structure)]` | 該幀偵測到的候選漣漪中心與結構分數 |

第一幀沒有前一幀可比，回傳 `None`。

### 活動等級（`activity_level(score)`）

```
score < 0.15 → idle      (灰)
score < 0.30 → low       (青)
score < 0.50 → medium    (黃)
score < 0.70 → high      (橘)
else         → intense   (紅)
```

> 建議 headline 用 `mid_score.max()`（最熱 cell），不是 `mid_score.mean()`。後者會被空水區稀釋。

---

## 5. 配置（`RippleAlgoConfig`）

| 類別 | 參數 | 預設 | 意義 |
|---|---|---|---|
| 前處理 | `blur_kernel`, `clahe_clip`, `clahe_tile` | 5 / 3.0 / 8 | |
| 反光 | `use_specular_mask`, `specular_v_thresh`, `specular_s_thresh`, `specular_dilate` | True / 220 / 50 / 5 | HSV 反光抑制 |
| MOG2 | `bg_history`, `bg_var_threshold`, `bg_dilate` | 500 / 16 / 5 | |
| MOG2 自適應 | `bg_learning_rate`, `bg_freeze_activity`, `bg_soft_floor` | −1 / 0.30 / 0.30 | 活動高時凍結；bg 區降權 |
| Canny | `canny_low`, `canny_high` | 25 / 80 | |
| 徑向環 | `r_min`, `r_max`, `radius_bins`, `bin_topk`, `bin_min_edges` | 8 / 120 / 8 / 2 / 6 | Radius-bin 設定 |
| 光流 | `flow_scale`, `dis_preset`, `flow_mag_threshold`, `divergence_cap` | 0.5 / 1 / 0.12 / 0.20 | |
| 動態 / 候選 | `diff_threshold`, `candidate_close_kernel`, `candidate_min_area`, `candidate_max_count`, `candidate_sigma`, `candidate_min_score` | 6 / 5 / 25 / 16 / 18 / 0.05 | |
| Soft cap | `motion_cap`, `edge_cap` | 0.03 / 0.03 | 軟飽和半值點 |
| 網格 | `grid_rows`, `grid_cols` | 8 / 8 | |
| 時序 | `short_window`, `mid_window`, `long_window` | 30 / 150 / 900 | |
| 融合 | `w_spatial`, `w_expansion`, `w_radial_flow`, `w_divergence`, `w_motion`, `w_edge` | 0.6/0.4 0.6/0.4 0.5/0.5 | |
| 結構合成 | `structure_combine` | `"linear"` | `"linear"` 或 `"geomean"` |
| 輸入縮放 | `input_scale` | 1.0 | 1920×1080 建議 0.5 |
| ROI | `roi_mask` | None | (H, W) uint8 0/1 遮罩 |

---

## 6. 性能設計

| 優化 | 效果 |
|---|---|
| Canny / Sobel 整張只算一次 | 共用，免重算 |
| DIS 光流（vs Farneback） | CPU 上 3-5× 加速 |
| 光流在 `flow_scale=0.5` 降採後計算 | 額外 ~4× 加速 |
| Candidate 模式（top-N，不是 64 cells） | 評分次數從 64 → ≤16 |
| Gaussian patch 預建，max-blend | 不用 cv2.GaussianBlur 整圖 |
| Grid pool 用切片，不是 per-pixel iterate | |
| `motion_density` / `edge_density` 用單次 mean | |
| Cell window deque + 直接 mean | 簡單但夠快；可升級為 vectorised running sum |

實測 540p × 8×8 grid：~13–16 fps（CPU），含 IO 與標註寫檔。

---

## 7. 主類別 API

```python
from core.ripple_algo import RippleDetector, RippleAlgoConfig, activity_level

detector = RippleDetector(RippleAlgoConfig(
    input_scale=0.5,
    grid_rows=8, grid_cols=8,
    structure_combine="linear",      # 或 "geomean" 求嚴格
))

for frame in video_stream:
    result = detector.process(frame)
    if result is None:
        continue

    peak = float(result.mid_score.max())
    label = activity_level(peak)

    overlay = cv2.applyColorMap(
        (result.heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    vis = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)

    for cx, cy, score in result.candidates:
        cv2.circle(vis, (cx, cy), 6, (255, 255, 255), 1)
```

`detector.reset()` 清空所有狀態（前一幀、滑動窗口、MOG2、DIS、cache）。

---

## 8. CLI

```
uv run main.py data/sample.mov --input-scale 0.5
uv run main.py data/sample.mov --no-show --output result.mp4
uv run main.py data/sample.mov --grid 8 --short-window 30 --mid-window 150
```

詳見 [main.py](../main.py)。

按 ESC 結束預覽視窗。輸出影像包含：

- JET 熱力圖 overlay
- 每個 cell 中心一個小點，依 mid_score 等級著色
- 每個候選點一個白色圈與 structure 分數
- 左上角文字：`peak / mean / level / frame`

---

## 9. v2 已知限制與下一步

| 問題 | 影響 | 可能改善 |
|---|---|---|
| 沒有相機穩定（ECC affine） | 風震、無人機抖動會被當作動態 | 加入 `cv2.findTransformECC` 把 prev_gray warp 到 current |
| 沒有透視校正 | 攝影機斜視時圓波在影像中是橢圓 | 加 homography（需四角標定）或橢圓 radial template |
| 沒有時序 ridge 偵測 | 反光偶爾一幀像圓，但不會穩定向外擴張 | 對候選點做 polar transform，跨幀堆 `S(t, r)` 找斜線 |
| Cell window 用 deque + mean | 64×3 deque 在大 grid 下可能變慢 | 改為 vectorised running sum，O(1) per frame |
| Candidate 上限 `candidate_max_count` | 大規模餵食時可能漏掉部分波紋 | 自適應 top-N（依 active 像素總量） |

---

## 10. 檔案 / 模組對照

| 區塊 | 函式 | 位置 |
|---|---|---|
| 設定 | `RippleAlgoConfig` | [core/ripple_algo.py](../src/core/ripple_algo.py) |
| 結果 | `ActivityResult` | 同上 |
| 前處理 | `_preprocess` | 同上 |
| 反光 mask | `_specular_mask` | 同上 |
| 邊緣 / 梯度 | `_canny_edges`, `_sobel_gradients` | 同上 |
| 散度 | `_compute_flow_divergence` | 同上 |
| 徑向 (legacy) | `_radial_edge_score`, `_radial_expansion_score` | 同上 |
| 徑向 (v2) | `_radius_bin_peak_edge_score`, `_ring_pool_mean` | 同上 |
| 候選點 | `_find_candidate_centers` | 同上 |
| Heatmap render | `_render_candidate_heatmap`, `_grid_max_pool`, `_grid_density_pool` | 同上 |
| Soft helper | `_soft_saturate` | 同上 |
| 主類別 | `RippleDetector` | 同上 |
| 等級標籤 | `activity_level` | 同上 |
| CLI 入口 | `run_on_video` | 同上 |
