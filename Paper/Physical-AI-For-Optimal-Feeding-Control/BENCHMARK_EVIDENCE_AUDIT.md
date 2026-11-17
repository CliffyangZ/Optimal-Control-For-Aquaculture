# Benchmark 證據核對與稿件修訂

核對日期：2026-11-05。來源：使用者指定的 `ssh MyLinux`，專案 `/home/cliffyang/Program/Aquaculture_CV_Project`。遠端專案只讀；沒有重訓、重跑偵測或覆寫任何 metrics、manifest、prediction、checkpoint。

**結論：測試集身分、來源標註、訓練設定與保存的偵測指標已能追溯和重現。但已確認目前切分不具影片群組互斥性，不能聲稱無資料洩漏或具有新影片泛化能力。**

## 1. 原始證據與保存位置

- [原始證據快照](audit/benchmark_evidence/)：benchmark 設定與程式、來源標註、資料集版本說明、split manifest、六組 run records／metrics／predictions、YOLO args／訓練 CSV、RF-DETR training_config／metrics CSV、相關套件程式。
- [擷取收據](audit/benchmark_evidence/capture_receipt.json)：來源檔案 SHA-256、512 張影像的實際雜湊核對、解碼／標註檔檢查、初始／最終權重雜湊，以及稽核時環境版本。未複製原始影像或大型權重。
- [補充收據](audit/benchmark_evidence/supplement_receipt.json)：YOLO checkpoint 自帶的版本與訓練參數等中繼資料。
- [重算結果](audit/recomputed_metrics.json)：每組 AP@300、對照 AP@100、P/R/F1 與 2,000 次 bootstrap 的完整精度值，並逐欄比較保存的 metrics。
- [機器可讀核對結果](audit/benchmark_audit.json)：檔案完整性、來源標註與 ground truth 一致性、manifest／run hashes、CSV／metrics／稿件數字一致性，以及群組重疊的獨立科學限制。
- [舊摘要](audit/legacy_model_comparison.md) 與 [修訂前稿件](audit/legacy_manuscript.tex) 保留供追溯；目前正文與圖表依原始 AP@300 紀錄更新。

來源 Git HEAD 為 `57b27e538e2d0a91709bc45e22532327182a896e`，但 `Research/` 和 `results/` 為 untracked。**這個 commit 無法單獨重建 benchmark**；因此保留實際讀取的程式及其雜湊，不能只用 run record 中的 `eval_commit=57b27e5` 作為完整版本證據。

## 2. 測試 manifest、切分與標註

資料集版本由來源匯出 metadata 確認：FishTracker_v2 **v1**，199 張，2026-05-17 匯出；feeding_fish **v2**，313 張，2026-05-19 匯出。兩者分別先以 stretch 轉為 640×640、512×512；metadata 記錄無匯出時增強。

六組 run 共享語意 split hash：

```text
c82ee4d4bf173c77a76da93843833d4f6cd63179db5bb4d7ed683ede56314def
```

| Split | FishTracker_v2 | feeding_fish | 影像 | 框 |
|---|---:|---:|---:|---:|
| train | 123 | 191 | 314 | 15,707 |
| val | 38 | 61 | 99 | 4,136 |
| heldout_real | 38 | 61 | 99 | 4,624 |
| 合計 | 199 | 313 | 512 | 24,467 |

已從來源標註重新計數，確認正文各類別框數與含該類別影像數，包括一張沒有標註框的訓練影像。512 張來源影像全部可解碼，實際 SHA-256 全部符合 manifest，標註檔全部存在；split 之間沒有相同影像位元組或完全相同解碼像素。

**但影片群組有重疊。** `data/split.py::make_split` 在 `(source_dataset, group)` 內打散影像，再分到三個 split；不是將整個群組分配到單一 split。八個由檔名推導的群組全部跨三個 split：

| 來源／檔名群組 | train | val | heldout_real |
|---|---:|---:|---:|
| feeding_fish / Timeline-1_mov | 7 | 1 | 1 |
| feeding_fish / Timeline-2_mov | 184 | 60 | 60 |
| fishtracker_v2 / downCamera_1-mp4 | 37 | 12 | 12 |
| fishtracker_v2 / downCamera_5-mp4 | 28 | 8 | 8 |
| fishtracker_v2 / downCamera_7 | 2 | 1 | 1 |
| fishtracker_v2 / downCamera_8-mp4 | 32 | 10 | 10 |
| fishtracker_v2 / frame | 9 | 2 | 2 |
| fishtracker_v2 / vlc-record-… | 15 | 5 | 5 |

因此已將正文改為 **same-video image holdout**，移除能被解讀為影片互斥的敘述。影像雜湊互斥不排除相鄰影格或近重複場景；也沒有把檔名群組當成經人工確認的八支獨立影片。

## 3. 訓練設定：區分預設與實際紀錄

共用上限 100 epochs、patience 20、有效 batch 16；訓練／切分 seed 42。以 run records、YOLO `args.yaml`／checkpoint 參數與 RF-DETR `training_config.json` 為準，沒有把 benchmark.yaml 的預設 micro-batch 當成實際值。

| Run | 輸入 | micro-batch × accumulation | 訓練紀錄 epochs | 選取權重 |
|---|---:|---:|---:|---|
| Y-n | 640 | 16 × 1 | 100 | ultralytics/weights/best.pt |
| Y-s | 640 | 16 × 1 | 100 | ultralytics/weights/best.pt |
| Y-m | 640 | 8 × 2 | 100 | ultralytics/weights/best.pt |
| R-n | 384 | 1 × 16 | 49 | rfdetr/checkpoint_best_total.pth |
| R-s | 512 | 1 × 16 | 88 | rfdetr/checkpoint_best_total.pth |
| R-m | 576 | 1 × 16 | 97 | rfdetr/checkpoint_best_total.pth |

RF-DETR epochs 依訓練 CSV 零起算最大 epoch + 1 記錄。YOLO 為 SGD，LR 0.01、momentum 0.937、weight decay 0.0005、warm-up 3、`cos_lr=false`、`lrf=0.01`。RF-DETR 的 AdamW 實作與設定已查閱，base LR 0.0001、encoder LR 0.00015、weight decay 0.0001、step scheduler、`lr_drop=100`、warm-up 0。

YOLO 增強參數可由保存的 args 核對；RF-DETR 保存 `aug_config=null`、multiscale／expanded_scales 啟用，套件預設只有水平翻轉 p=0.5 的增強配置。原先未實驗的 D-FINE 欄位已移出正文。兩族系使用不同原生 checkpoint 選取機制；RF-DETR 比較 regular／EMA 的原生 validation AP，且其 validation `eval_max_dets=500`，不能混同最終共用的 AP@300。這些差異使結果屬 pipeline 比較。

YOLO 權重記錄 Ultralytics 8.4.53。稽核環境為 RF-DETR 1.7.0、PyTorch 2.12.0、torchvision 0.27.0、numpy 2.4.6、pycocotools 2.0.11；完整版本在擷取收據中。稽核時安裝版本不等於歷史環境已被獨立封存。

## 4. 評估流程與舊摘要差異

六組 `metrics.json` 都記錄 `eval_set=heldout_real`、99 張影像、4,624 個框；所有 predictions 覆蓋相同 0–98 的影像 ID，依 manifest 順序對應。完整 detection 區塊（含 CI）由保存的 predictions 和重新建立的 ground truth 重算後，與 metrics 在 1e-10 容差內一致。

- Predictions：score floor 0.001、每張最多 300 筆。
- AP：pycocotools，IoU 0.50:0.05:0.95，`maxDets=[1,10,300]`，從累積陣列的 300 欄取值；overall 為四類平均。
- P/R/F1：confidence 0.25、IoU 0.5、分類別依 confidence 由高到低的一對一 greedy matching，合併 TP/FP/FN 後計算。
- CI：image-level percentile bootstrap，B=2,000、seed=123；未涵蓋影片相依及訓練 seed 變異。

舊摘要的所有 overall/feed/fish AP50–95 都與同批 predictions 的 **AP@100** 重算值在三位小數一致；目前遠端 metrics／CSV 則用 **AP@300**。修訂統一採 AP@300，而非混用或挑選較高數值。

| 指標 | 舊摘要 AP@100 | 核對後 AP@300 |
|---|---:|---:|
| Y-s overall AP50–95 | 0.254 | 0.271 |
| Y-m feed AP50–95 | 0.186 | 0.258 |
| R-s overall AP50–95 | 0.354 | 0.358 |
| R-m overall AP50–95 | 0.357 | 0.363 |

另有一處舊摘要捨入／轉錄差異：Y-n overall AP50 原寫 0.416，但 AP@100 重算為 0.416506，三位小數應為 0.417。新版使用原始 AP@300 的 0.456，不沿用此舊值。

已同步更新摘要、結果表、兩張比較圖、模型選擇段落、結論及 `assets/model_comparison.md`。圖表由保存的 metrics 產生，F1 增加對應的影像 bootstrap CI；AP 沒有虛構 CI。

## 5. 時間與記憶體證據的實際限度

保存的六組 metrics 的 `system.note` 都說明該區塊是 **從 evaluation_table.csv 回復**，完整歷史 profile 區塊未保留。本次未重新量測 GPU 延遲，不能將數字一致誤寫成延遲已實測重現。

已查閱 profiler，確認每次 timed iteration 後 CUDA synchronization、50 warm-up／200 iterations、單張測試影像、batch 1、model-only FP32 tensor，以及 `max_memory_allocated()/2**20`。所以記憶體單位修正為 **MiB 的 peak allocated memory**，不是 nvidia-smi 整張顯卡的記憶體占用。

兩個 pipeline 的起始影像格式不同：YOLO 是 BGR array；RF-DETR 的 RGB/PIL 轉換在計時外。正文已揭露；p50 倒數改名為 derived rate，不再等同實測 streaming FPS。原始 GPU clocks／power-state 無法由回復的 system 區塊確認，已移除確定性描述。

## 6. 架構一致性

依現有架構圖與時序指南統一為：**server 執行偵測、品質檢查、Kalman 與決策狀態機；Pi 擷取／串流並轉送命令；BMduino 控制伺服馬達。** 正文、圖說及時序指南已一致。

此為參考設計配置，沒有聲稱經部署 trace 驗證。伺服器失聯時 Pi 本地命令逾時停止續餵，明列為待實測的需求。

## 7. 重現方式與仍待完成事項

在本工作區重跑 artifact 核對與圖表：

```bash
python3 scripts/audit_benchmark_evidence.py > audit/benchmark_audit.json
MPLCONFIGDIR=/tmp/feeding-review-matplotlib python3 scripts/render_benchmark_figures.py
latexmk -xelatex -interaction=nonstopmode -halt-on-error PhysicalAI_For_Optimal_Feeding_Control_6p.tex
```

需要重新從原始主機擷取／重算時，兩個 remote script 都只讀取專案，以 stdout 回傳結果，且應設定 `PYTHONDONTWRITEBYTECODE=1`；不要執行會覆寫 run metrics 的 benchmark `evaluate` CLI 代替本次核對。

尚不能由本次修訂完成的科學驗證：確認真實影片／session 分組後做 group-disjoint 切分，重新訓練六組模型並在未參與訓練／選模的影片上測試；保留完整 profiling 設定與原始時間序列；驗證時序訊號與投餌閉環。保存的 predictions 未嵌入 checkpoint／split 雜湊，本次核對其來源關係依 run record 與可重現的指標，不等同由 checkpoint 重新執行 inference。
