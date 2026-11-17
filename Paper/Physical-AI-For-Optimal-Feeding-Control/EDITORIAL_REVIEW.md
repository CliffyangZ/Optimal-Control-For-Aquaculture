# 論文撰寫與前後一致性修訂

日期：2026-11-05。修訂對象為 `PhysicalAI_For_Optimal_Feeding_Control_6p.tex` 及其比較圖繪製腳本。已核對的資料表數值、AP@300 協定與原始研究紀錄保持一致。

| 問題 | 修正 |
|---|---|
| Introduction 與 Related Work 重述同一批文獻與方法動機 | Introduction 聚焦問題、方法定位與證據範圍；Related Work 負責比較回饋訊號與偵測器族系。 |
| 摘要、結果、模型選擇與結論反覆列出相同數字 | 結果段解讀模型排名，選模段解釋 feed F1／延遲取捨；結論改為回答研究能支持的判斷。 |
| 設計規格與已完成實驗使用相同的肯定語氣 | 統一區分 proposed/reference design、已重算的偵測 benchmark、未重測的歷史 GPU 時間及待進行的控制／魚缸驗證。 |
| 一處說濾波與品質參數已校正，另一處又說校正待做 | 全文改為需要獨立校正，未補入任何虛構參數。 |
| 跨裝置事件被描述為共用 monotonic clock | 改為 Pi 時間基準，其他裝置需時鐘對應與誤差紀錄，或以 Pi 收發時間表示。 |
| 圖說寫 ROI 內面積，公式卻對相交框計入完整面積 | 統一說明固定相機像素座標、反轉模型縮放、相交框完整面積與重疊偏差；未修改原始面積定義。 |
| 無效影格只做預測，但更新公式未說明例外 | 明列有效觀測才更新 posterior；缺失／拒絕位置保留 prediction，且不算有效樣本。 |
| 消失比例被稱為速率，並與另一套 log/frame 飢餓分區並列 | 統一使用固定視窗的 relative disappearance ratio，定義 epsilon 的用途；舊飢餓圖移出正文，保留 `assets/hunger Level.png`。 |
| baseline、增加資格與安全限制的語意不清 | baseline 寫為待校正的 nominal portion；保留兩個連續合格視窗的增加條件，說明低初始面積不得增加。將「防止過餵」改為命令限制的設計目的。 |
| Suspend 與非零最小份量可能被解讀為矛盾 | 份量上下界只適用主動出料；Suspend 不發送出料命令。 |
| Remaining System Validation 與 Discussion 重複限制與未來工作 | 合併為 Discussion and Validation Priorities，依證據範圍、訊號／部署與魚缸結果分段。 |
| 複合圖缺少子圖編號 | 比較圖補 (a)/(b)/(c)；準確度－延遲圖補 (a)/(b)，圖說與正文分別引用對應面板。 |
| YOLO26 缺正式方法引用；部分縮寫未展開 | 補入已核對的 [YOLO26 原始論文](https://arxiv.org/abs/2606.03748)，展開 RTS、EMA、FCR。 |

移除舊 Fig. 5 後，原準確度圖 Fig. 6 自動改為 **Fig. 5(a–c)**；原準確度－延遲圖 Fig. 7 改為 **Fig. 6(a–b)**。交叉引用使用 LaTeX labels，沒有手工修改硬編號。

本次新增的 no-fish 對照、決策誤差與故障測試敘述均屬待完成的驗證需求，未寫成已取得的結果。資料相依性與缺少控制實驗的限制也沒有以文字修訂取代。

核對項目：66 項原始證據／稿件數字一致性檢查通過；無重複 labels、未解析引用、缺失文獻或未引用的 bibliography entries。正文排除圖表、公式與 bibliography 後，未發現長度至少 80 字元的完全相同重複句；另以人工檢查語意重複與跨節銜接。PDF 編譯為六頁並逐頁檢查。
