# Web Backend and Frontend for RF-DETR Aquaculture Monitoring

## Architecture

The web backend uses FastAPI to expose:
- **WebRTC video stream** via MediaMTX WHEP (HTTP Egress Protocol) for low-latency video delivery
- **Real-time detection overlays** via WebSocket (`/ws/overlays`) using the `detections.v1` schema
- **Feed analysis and hunger level** via WebSocket analytics messages
- **Session lifecycle management** via HTTP REST endpoints

## Detection Backend

The approved YOLO object detection approach has been superseded by RF-DETR, which offers superior accuracy for fish and feed area detection. The WebSocket messages still carry detection data compatible with the previous `detections.v1` schema for backward compatibility.

## WebSocket Messages

The `/ws/overlays` endpoint broadcasts three message types:

### detections.v1
Real-time bounding-box detections with normalized coordinates [0.0, 1.0]:
```json
{
  "schema": "detections.v1",
  "type": "detections",
  "source_id": "cam",
  "frame_id": 42,
  "timestamp_ms": 1700000000000,
  "source_width": 1280,
  "source_height": 720,
  "stream_status": "live",
  "detection_status": "active",
  "detections": [
    {"class_name": "fish", "confidence": 0.92, "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
  ]
}
```

### analytics.v1
Feed area analysis and hunger level assessment (updated every 2s):
```json
{
  "schema": "analytics.v1",
  "type": "analytics",
  "source_id": "cam",
  "timestamp_ms": 1700000000000,
  "hunger_level": "Mild",
  "consumption_rate": -0.03,
  "frame_ids": [40, 41, 42],
  "raw_area": [1200.0, 1150.0, 1050.0],
  "kalman_area": [1200.0, 1175.0, 1100.0],
  "ema_area": [1190.0, 1170.0, 1120.0]
}
```

### status.v1
Stream and detection health status:
```json
{
  "schema": "status.v1",
  "type": "status",
  "source_id": "cam",
  "timestamp_ms": 1700000000000,
  "stream_status": "live",
  "detection_status": "active",
  "active_viewer_count": 2,
  "message": "System healthy"
}
```

## Stream Health States

### StreamStatus
- `waiting`: Session created, no frames received yet
- `connecting`: First frame acquisition in progress
- `live`: Receiving frames within normal interval
- `reconnecting`: Stream interrupted, attempting recovery
- `stream_unavailable`: No frames received for > 3s (INTERRUPTION_TIMEOUT_SECONDS)
- `degraded`: Stream health degraded but not unavailable
- `error`: Fatal stream error

### DetectionHealthStatus
- `loading`: RF-DETR model loading
- `active`: Detections active and recent
- `stale`: `last_overlay_at` older than `stale_overlay_ms` (3000ms)
- `unavailable`: No model loaded or detection error

The `stale_overlay_ms`, `overlay_ws_url`, and `last_overlay_at` fields are key to understanding detection freshness and health monitoring.

## HTTP Endpoints

### GET /config
Returns WebRTC WHEP URL and WebSocket configuration:
```json
{
  "source_id": "cam",
  "webrtc_url": "http://127.0.0.1:8889/cam/whep",
  "overlay_ws_url": "ws://127.0.0.1/ws/overlays",
  "stale_overlay_ms": 3000,
  "local_trusted_network_only": true
}
```

### GET /status
Current session snapshot (stream and detection status, viewer count).

### POST /session/start
Register a viewer and start the detection worker (if first viewer).

### POST /session/stop
Unregister a viewer and stop the worker (if last viewer after idle delay).

## Environment Variables

```bash
STREAM_URL=rtsp://192.168.137.157:8554/cam
SOURCE_ID=cam
MEDIAMTX_HOST=192.168.137.157
MEDIAMTX_WEBRTC_PORT=8889
RFDETR_MODEL_PATH=models/rf-detr-fish-regular.pth
RFDETR_DEVICE=auto              # "auto", "cuda", "cpu", "mps"
RFDETR_CONFIDENCE_THRESHOLD=0.25
STALE_OVERLAY_MS=3000
TARGET_DETECTION_FPS=1
```

## Development

Backend (with model):
```bash
export RFDETR_MODEL_PATH=/path/to/model.pth
uv run --group ml uvicorn web.app:app --reload
```

Frontend (Vite dev server with API proxy):
```bash
cd src/web/frontend
npm install
npm run dev
```

The Vite dev server proxies `/config`, `/status`, `/session`, and `/ws` to the backend at `:8000`.
