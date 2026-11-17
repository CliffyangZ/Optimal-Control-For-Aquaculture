# 模型比較訓練方法與指令

對象：要自己執行訓練的研究者。範圍：D-FINE／RF-DETR／YOLO26 × n／s／m，共 9 組 run 的**訓練**部分（評估、報表尚未實作）。

規格與設計見 `specs/006-model-comparison-benchmark/`（`spec.md`、`plan.md`、`tasks.md`），比較矩陣與評估協定見根目錄 `Training_Method.md`。

---

## 1. 做了什麼

| 項目 | 內容 |
| --- | --- |
| 資料 | `fishtracker_v2`（4 類）+ `feeding_fish`（2 類），合併後 512 張 |
| 類別 | `debris`、`feed`、`fish`、`lumpfish`；`pellet→feed`、`feeding-fish→fish` |
| 切分 | 鎖定一次：train 314／val 99／heldout_real 99（seed 42，依來源與影片群組分層） |
| 訓練預算 | 所有 run 相同：最多 100 epochs、patience 20、有效 batch 16、imgsz 640 |
| 預訓練權重 | `models/pretrained/`（見該目錄 `README.md`） |
| 輸出 | `Results/benchmark/runs/<run>/` |

三個族系都用各自原生的訓練器，之後再以同一份評估程式比較：

| 族系 | Run id | 訓練器 | 備註 |
| --- | --- | --- | --- |
| YOLO26 | `Y-n` `Y-s` `Y-m` | Ultralytics | 增強與優化器參數讀自 `Depolyment/src/config/training.yaml` |
| RF-DETR | `R-n` `R-s` `R-m` | `rfdetr` 套件 | **不使用 640**，用官方解析度：n=384、s=512、m=576（記在 run record） |
| D-FINE | `D-n` `D-s` `D-m` | 自寫 Hugging Face 訓練迴圈 | AdamW、bf16、cosine LR；以 val AP@[.5:.95] 選最佳權重 |

有效 batch 固定 16；8 GB 顯卡的小檔 micro-batch 用梯度累積補足（`micro_batch × grad_accum = 16`）。

---

## 2. 環境

```bash
cd /home/cliffyang/Program/Aquaculture_CV_Project
export PYTHONPATH=Research/benchmark/src
PY=.venv/bin/python
```

W&B（預設開啟）：訓練前先登入一次；未登入時 `train` 會在寫入任何紀錄前直接中止（結束碼 2）。

```bash
.venv/bin/wandb login        # 或 export WANDB_API_KEY=...
```

需求：`ml` 依賴群組已安裝（含 `ultralytics`、`rfdetr[train,loggers]`、`transformers`、`pycocotools`；RF-DETR 訓練需要 `[train,loggers]` extras）、`models/pretrained/` 內九組權重齊全。缺權重時：

```bash
.venv/bin/python scripts/download_pretrained.py
```

---

## 3. 指令

### 3.1 鎖定切分（只做一次）

```bash
$PY -m benchmark.cli split
```

產生 `Results/benchmark/splits/split_manifest.json` 與 `class_map.json`。已存在時**拒絕覆蓋**（結束碼 3）；只有在確定要重來並放棄所有舊 run 時才加 `--force`。

### 3.2 訓練

建議順序：先跑三個 `s` 驗證流程（對應 `tasks.md` T032），確認沒問題再補 `n`、`m`。

```bash
# 1) 管線驗證
$PY -m benchmark.cli train --run Y-s --run R-s --run D-s

# 2) 其餘六組
$PY -m benchmark.cli train --run Y-n --run R-n --run D-n
$PY -m benchmark.cli train --run Y-m --run R-m --run D-m
```

選項：

| 選項 | 說明 |
| --- | --- |
| `--run <id>` | 可重複；`<id>` 為 `Y-s`、`R-m`、`D-n` 等；第二個 seed 用 `Y-s-seed7` |
| `--micro-batch N` | 顯存不足時改小 micro-batch（累積步數自動增加，有效 batch 不變） |
| `--force` | 重訓已完成（`trained`／`evaluated`）的 run |
| `--config <path>` | 指定其他設定檔（放在 `split`／`train` 之前） |

例：

```bash
$PY -m benchmark.cli train --run R-m --micro-batch 2          # OOM 時
$PY -m benchmark.cli train --run Y-s-seed7                    # 第二個 seed
$PY -m benchmark.cli --config my.yaml train --run D-s
```

**OOM 時只降 micro-batch，不要降 imgsz**（輸入尺寸是控制變因）。

結束碼：`0` 成功、`2` 設定或輸入錯誤（如缺權重、尚未 split）、`3` 會覆蓋已鎖定檔案、`4` 訓練失敗。

### 3.3 單元測試

```bash
.venv/bin/python -m pytest Research/benchmark/tests -q
```

---

## 4. 設定檔

`Research/benchmark/configs/benchmark.yaml`：

| 區塊 | 用途 |
| --- | --- |
| `seed`、`split` | 隨機種子與切分比例（改動會改變切分，需 `--force`） |
| `budget` | `max_epochs`、`patience`、`effective_batch`、各族系各檔位的 `micro_batch` |
| `imgsz` | YOLO26／D-FINE 輸入尺寸（RF-DETR 不受影響） |
| `tracking.wandb` | 預設 `enabled: true`、專案 `Aqua_CV_Project`；`entity: null` 用帳號預設；`enabled: false` 才不記錄 |
| `runs` | 九組 run 與預訓練權重路徑 |
| `thresholds`、`bootstrap`、`decision` | 評估階段用，訓練不使用 |

所有 run 共用同一份設定；**不要為單一模型調整訓練參數**，否則比較不公平。要改共用預算（例如 epoch 上限）請改設定檔後，把三個 `s` 重跑。

---

## 5. W&B 紀錄

三個族系用同一套命名，方便在 W&B 並排比較：

| 欄位 | 值 |
| --- | --- |
| project | `Aqua_CV_Project` |
| run name | run id（`Y-s`、`R-m-seed7` …） |
| group／tags | 族系（`yolo26`／`rfdetr`／`dfine`）、檔位、`seed<k>` |
| config | seed、budget、micro-batch／累積步數、imgsz、`split_hash`、`class_map_hash`、`config_hash`、評估程式 commit |
| 指標 | YOLO26、RF-DETR 由各自訓練器記錄；D-FINE 每 epoch 記 `train_loss`、`val_ap`、`val_ap50` |

W&B run id 會寫進 `run-record.json` 的 `wandb_run_id`。RF-DETR 的 run 由 `rfdetr` 套件自己建立，若讀不到 id，該欄位會留空，可到 W&B 以 run name 找。

---

## 6. 輸出結構

```text
Results/benchmark/
├── splits/
│   ├── split_manifest.json      # 鎖定的切分（含每張圖 SHA-256）
│   ├── class_map.json
│   └── export/                  # 各訓練器的資料格式（自動產生）
└── runs/<run>/
    ├── run-record.json          # 狀態、seed、權重路徑、wandb id、各種 hash、解析度
    ├── ultralytics/weights/best.pt      # Y-*
    ├── rfdetr/checkpoint_best_*.pth     # R-*
    ├── dfine/best/                      # D-*（Hugging Face 格式）
    └── train_error.log          # 僅失敗時
```

`run-record.json` 的 `status`：`pending → trained`，失敗為 `failed`（附 `failure_reason`）。失敗的 run 會保留紀錄，不會被默默替換。

---

## 7. 注意事項

- **資料洩漏**：`feeding_fish` 幾乎都來自同一支影片（299 張中 292 張），無法按影片切分，相鄰幀可能同時出現在 train 與 held-out，分數會偏樂觀。結論請以後續的合成飼料評估為準。
- **RF-DETR 解析度不同**：與其他兩族比較時需在報告註明（n=384、s=512、m=576）。
- **D-FINE 為自寫迴圈**：增強近似官方（左右翻轉、亮度對比、HSV、縮放平移）；超參數在 `train/dfine.py` 開頭的常數（`LR`、`WEIGHT_DECAY` 等）。
- **訓練耗時**：未實測；先跑一個 `s` 看每 epoch 時間，再決定是否縮減 `max_epochs`。
- **尚未實作**：預測轉換、評估（AP、P/R/F1 與 95% CI、計數與 Bland–Altman）、延遲量測、報表與選模規則，見 `specs/006-model-comparison-benchmark/tasks.md` 未勾選項目。
