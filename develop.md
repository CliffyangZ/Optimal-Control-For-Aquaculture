這是一個非常有挑戰性且具備實務價值的架構。要實現無人機視角下的水面波紋辨識，低延遲與穩定處理流是核心關鍵。
針對你的需求，我們將架構設計為三個主要層次：採集層（Smartphone）、傳輸交換層（MediaMTX）、以及處理展示層（Laptop/Streamlit）。
1. 系統架構圖 (MVP)
這個架構採用「生產者-消費者」模式，確保影像處理邏輯與介面顯示分離。
• 採集端 (Phone): 使用支持 RTSP 推流的 App（如 Larix Broadcaster），將手機鏡頭畫面推送到筆電。
• 中轉站 (MediaMTX): 作為後台服務，接收手機推流並轉發給 Python 腳本。
• 核心處理器 (Python + OpenCV): 負責從 MediaMTX 抓取 RTSP 流，進行波紋偵測運算。
• 顯示介面 (Streamlit): 呈現處理後的畫面與數據分析。
2. 技術棧建議
組件	技術選擇	說明
傳輸協議	RTSP (over UDP)	為了極致低延遲，建議優先使用 UDP 而非 TCP。
串流伺服器	MediaMTX	輕量化且效能極高，適合作為本地開發後台。
影像處理	OpenCV (cv2)	處理 Frame 的核心，後續可接 YOLO 或物理過濾演算法。
前端框架	Streamlit	快速搭建監控介面，建議搭配 webrtc 組件或 placeholder 更新畫面。
並發處理	Multi-threading	影像抓取 (Producer) 與 畫面處理 (Consumer) 必須分開執行。
3. 檔案規劃方式 (Folder Structure)
一個清晰的模組化結構能讓你在後續加入 AI 辨識模型時更輕鬆：
drone-vision-project/
├── mediamtx/           # 放置 mediamtx 執行檔與配置文件
│   └── mediamtx.yml    # 配置路徑與協定
├── src/
│   ├── __init__.py
│   ├── config.py       # 存放 RTSP URL, 解析度, 門檻值等
│   ├── stream_reader.py # 負責從 RTSP 抓取 Frame 的 Thread
│   ├── processor.py     # 影像演算法：波紋偵測、濾波、座標轉換
│   └── utils.py         # 繪圖工具、座標計算
├── app.py              # Streamlit 主程式入口
├── requirements.txt
└── README.md

4. 模組互動方式與優化邏輯
關鍵：解決 OpenCV 讀取 RTSP 的延遲問題
OpenCV 的 VideoCapture 默認會有緩存機制，當處理速度慢於推流速度時，畫面會出現明顯延遲甚至疊加。
建議方案：
1. 專屬讀取執行緒 (Threaded Capture): 建立一個專門讀取 Frame 的線程，永遠只儲存「最新的一幀」，舊的幀直接丟棄。
2. 緩存優化: 在 cv2.VideoCapture 中設定 CAP_PROP_BUFFERSIZE 為 0。
3. 解析度縮減: 水面波紋偵測初期不需要 4K 畫面，建議在手機推流端限制在 720p 或更低，以減少網路頻寬與運算負擔。
軟體組件溝通順序
1. 啟動 MediaMTX: 設定路徑（例如 rtsp://localhost:8554/drone_live）。
2. 手機推流: 將鏡頭影像推向該路徑。
3. stream_reader.py: 開啟一個背景線程，不斷更新全域變數 current_frame。
4. processor.py: 從 current_frame 拿取影像，進行灰階處理、高斯模糊、Sobel 邊緣檢測（偵測水波），計算出魚群可能位置。
5. app.py: Streamlit 定期從 processor 拿取處理後的影像與偵測數據，更新 UI。
5. MVP 開發的第一步建議
你可以先用這段簡易的 Python 結構來測試 OpenCV + MediaMTX 的連通性，這是優化延遲的基礎：
import cv2
import threading

class VideoStream:
    def __init__(self, rtsp_url):
        # 使用 FFMPEG 並強制 UDP 降低延遲
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = False, None
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def get_frame(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()