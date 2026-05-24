# Aquaculture_CV_Project

## 開啟後端與前端

### 後端

```bash
export YOLO_MODEL_PATH=/absolute/path/to/approved-model.pt
uv run --group ml uvicorn web.app:app --host 0.0.0.0 --port 8000
```

後端啟動後可開啟 `http://localhost:8000`。

### 前端

```bash
cd web/frontend
npm install
npm run dev
```

前端開發伺服器預設在 `http://localhost:5173`，API 會代理到 `http://localhost:8000`。

### File Structure

```text
Aquaculture_CV_Project/
├── config/
│   └── setting.py                              # Stream and approved YOLO runtime settings
├── core/
│   ├── infer/
│   │   └── detector.py                         # Approved YOLO detection inference
│   ├── models.py                               # Web session, stream, and detection statuses
│   └── stream_health.py
├── drivers/
│   └── README.md
├── streaming/
│   └── latest_frame_reader.py
├── tests/
│   ├── unit/
│   │   ├── test_overlay_payloads.py
│   │   └── test_module_boundaries.py
│   └── integration/
│       ├── test_yolo_detection_worker.py
│       └── test_monitoring_ui_contract.py
├── validation/
│   ├── clip_manifest.py
│   └── run_clip_validation.py
└── web/
    ├── app.py                                  # FastAPI backend
    ├── detection_worker.py                     # Shared detection loop
    ├── overlay_hub.py                          # WebSocket broadcast hub
    ├── schemas.py                              # Pydantic API/WS schemas
    └── frontend/                               # React/Vite UI
        ├── src/
        └── dist/                               # Built for production serving
```

## Hardware Components

### The Brain
- MPU (Rasperry Pi Zero 2W)
- MCU (BMduino)
- Camera (Rasperry Pi 5 IMX219 79.3 degree)


[Detection Model Google Drive Link](https://drive.google.com/drive/folders/13WuaHYUf2__m-g2ha3vrVkBRIhlYSeEh?usp=drive_link)
