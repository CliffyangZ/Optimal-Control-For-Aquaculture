# Aquaculture_CV_Project

## 開啟後端與前端

### 後端

```bash
export RFDETR_MODEL_PATH=/absolute/path/to/rf-detr-fish.pth
export STREAM_URL=rtsp://127.0.0.1:8554/cam
uv run --group ml uvicorn web.app:app --host 0.0.0.0 --port 8000
```

後端啟動後可開啟 `http://localhost:8000`。

### 前端

```bash
cd src/web/frontend
npm install
npm run dev
```

前端開發伺服器預設在 `http://localhost:5173`，API 會代理到 `http://localhost:8000`。

### Inferencing cmd
```bash
 uv run python train.py infer --model models/rf-detr-fish-regular.pth --source data/toilet_test.mp4 --model-type rfdetr --output results/outputs/test_toilets.mp4 --conf 0.3
```


### File Structure

```text
Aquaculture_CV_Project/
├── .github/
│   └── Assets/                                 # Architecture diagrams
├── Docs/
├── results/
│   ├── outputs/                                # Inference outputs
│   ├── runs/                                   # Training runs
│   └── wandb/                                  # Weights & Biases local logs
├── src/
│   ├── config/
│   │   └── setting.py                          # Stream and RF-DETR runtime settings
│   ├── core/
│   │   ├── infer/
│   │   │   └── rfdetr_detector.py              # RF-DETR detection inference
│   │   ├── models.py                           # Web session, stream, and detection statuses
│   │   └── stream_health.py
│   ├── drivers/
│   │   └── README.md
│   ├── pi_controller/                          # Raspberry Pi feed controller submodule
│   ├── streaming/
│   │   └── latest_frame_reader.py
│   ├── validation/
│   │   ├── clip_manifest.py
│   │   └── run_clip_validation.py
│   └── web/
│       ├── app.py                              # FastAPI backend
│       ├── detection_worker.py                 # Shared detection loop
│       ├── overlay_hub.py                      # WebSocket broadcast hub
│       ├── schemas.py                          # Pydantic API/WS schemas
│       └── frontend/                           # React/Vite UI
│           ├── src/
│           └── dist/                           # Built for production serving
└── tests/
    ├── unit/
    └── integration/
```

## YOLO Web Detection

The project previously used YOLO-based object detection via a WebSocket overlay with `detections.v1` schema. This is **not the active web detection behavior** — the current production system uses RF-DETR for improved accuracy and performance on the target use case.

## Hardware Components

### The Brain
- MPU (Rasperry Pi Zero 2W)
- MCU (BMduino)
- Camera (Rasperry Pi 5 IMX219 79.3 degree)


[Detection Model Google Drive Link](https://drive.google.com/drive/folders/13WuaHYUf2__m-g2ha3vrVkBRIhlYSeEh?usp=drive_link)
