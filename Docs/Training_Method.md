# 訓練模型選擇與比較

針對本專案目標——**水下／俯視場景的 fish + feed（pellet）偵測**，並以模型估計飼料數／面積對人工計數做一致性評估（Bland–Altman 等）。

## 比較範圍

族系（僅此三類）：

- D-FINE
- RF-DETR
- YOLO26

權重規模（僅 **n / s / m**）：共 **3 × 3 = 9** 組 run。

預訓練權重已下載至 `models/pretrained/`（見該目錄 `README.md`；重抓：`.venv/bin/python scripts/download_pretrained.py`）。

資料現況：`fishtracker_v2`（4 類，~199 張，`feed` 極多且小）、`feeding_fish`（2 類，313 張，valid 極小）。評估以 hold-out 影片／合成飼料為準，勿只盯 Roboflow valid mAP。

---

## 1. 權重規模對齊

| 檔位 | YOLO26 | D-FINE | RF-DETR | 角色 |
| --- | --- | --- | --- | --- |
| **n** | `models/pretrained/yolo26/yolo26n.pt` | `models/pretrained/dfine/dfine-n/` | `models/pretrained/rfdetr/rf-detr-nano.pth` | 邊緣／高 FPS |
| **s** | `…/yolo26s.pt` | `…/dfine-s/` | `…/rf-detr-small.pth` | 速度–精度平衡、現有 baseline |
| **m** | `…/yolo26m.pt` | `…/dfine-m/` | `…/rf-detr-medium.pth` | 精度較高、主比較檔 |

注意：

- RF-DETR 各檔參數量接近，主要差在 **輸入解析度**；與 YOLO26／D-FINE 比時務必同報 **latency / FPS @ 固定硬體**。
- D-FINE 建議用 COCO（或 Obj365→COCO）預訓練再 fine-tune；類別單純時留意過擬合。
- 不做 `l`／`x`／XL；資料量小，規模曲線停在 **m**。

---

## 2. 完整比較矩陣（9 runs）

同一 train／val split、同一 seed、同一 eval 協定。

| Run | 模型 | 權重 |
| --- | --- | --- |
| Y-n | YOLO26 | `yolo26n` |
| Y-s | YOLO26 | `yolo26s` |
| Y-m | YOLO26 | `yolo26m` |
| R-n | RF-DETR | Nano |
| R-s | RF-DETR | Small |
| R-m | RF-DETR | Medium |
| D-n | D-FINE | N |
| D-s | D-FINE | S |
| D-m | D-FINE | M |

產出兩張圖：

1. **同檔位橫比**（n／s／m 各一組）：三族系誰在 feed Recall／計數 MAE 較優  
2. **規模曲線**（各族系 n→s→m）：精度 vs latency Pareto

**預測（實驗前）：**

| 現象 | 預期 |
| --- | --- |
| n→s | feed 小物件 recall 明顯升 |
| s→m | 再升一截；多數場景 CP 可能落在 s 或 m |
| 同檔 DETR vs YOLO | RF-DETR／D-FINE 在密集 pellet 上計數可能較穩；YOLO26 訓練快、FPS 易拉高 |
| RF-DETR vs D-FINE | 皆無 NMS；若 D-FINE 無明顯計數優勢，部署可只留已接好的 RF-DETR |

部署選點：

- **Edge／RTSP**：滿足最低 FPS 的最小檔（多半 n 或 s）
- **Server／研究**：同約束下計數 MAE 最低者（多半 s 或 m）

---

## 3. 控制變因

| 項目 | 規範 |
| --- | --- |
| 資料 | 同一 `data.yaml`／class map；切分鎖定 |
| 輸入尺寸 | YOLO26／D-FINE 統一（建議先 640；小物件再掃 832）；RF-DETR 用該檔官方解析度並標明 |
| 訓練預算 | 同 epoch 上限 + early stopping（同 patience） |
| 增強 | YOLO26 用現有 `training.yaml`；DETR 用各自官方增強並註明 |
| 後處理 | YOLO：固定 conf／IoU／max_det；DETR：固定 conf／query 上限；禁止單模型刷榜調參 |
| 硬體 | 固定同一推理裝置；報 batch=1 latency 與串流 FPS |
| 隨機性 | 固定 seed；關鍵 run 建議 2 seeds（mean±std） |

類別對應（合併時）：`feed` ↔ `pellet`；`fish` ↔ `feeding-fish`；其餘保留或 ignore（實驗前寫死）。

---

## 4. 預測評估協定

### 4.1 Detection

| 指標 | 用途 |
| --- | --- |
| mAP50、mAP50-95（overall + per-class） | 總體品質 |
| feed／fish Precision、Recall、F1 | 投餵控制主偵測指標 |
| Confusion matrix | debris ↔ feed |
| AP_S 或面積分桶 AP | 小 pellet |
| 定性抽樣 | 高密度／低光／遮擋 |

### 4.2 飼料量預測（主 KPI）

| 指標 | 定義 |
| --- | --- |
| 計數誤差 | \(\hat{n}\) vs 人工：MAE、RMSE、ρ |
| 面積誤差 | \(\sum\) bbox 面積 vs 人工 |
| Bland–Altman | 均值差、95% LoA；分模型 |
| 95% CI | 誤差／偏差區間 |
| 校準 | 可選 \(\hat{n}'=a\hat{n}+b\)，報校正前後 |

聚合方式全模型固定（例：1 FPS 抽樣，或 clip median／sum）。

### 4.3 系統

Latency p50／p95、有效 FPS（含前／後處理）、VRAM peak。

### 4.4 決策

1. **研究優勝**：hold-out（真實＋合成）feed 計數 MAE 最低，且 Bland–Altman 無嚴重比例偏差。  
2. **上線**：目標硬體達最低 FPS 下，feed Recall 達門檻且 MAE 最小。  
3. **淘汰**：同檔 feed Recall 明顯落後，或 train 高、hold-out 崩（過擬合）。

---

## 5. 執行順序

1. 跑滿 9 runs（可先 Y-s／R-s／D-s 驗證管線，再補 n／m）  
2. 畫同檔橫比 + n→s→m Pareto  
3. 選 1 個 edge + 1 個 server checkpoint → 完整 §4（含 Bland–Altman）  
4. 記錄 wandb、權重路徑、eval 腳本版本、class map

**一句話：** 只比 D-FINE／RF-DETR／YOLO26 的 n／s／m 共九組；用飼料計數一致性與延遲選模，不看整體 mAP alone。
