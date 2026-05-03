"""MVP configuration constants for low-latency ripple monitoring."""

STREAM_URL = "rtsp://127.0.0.1:8554/drone_live"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_DISPLAY_FPS = 30
TARGET_DETECTION_FPS = 10
INTERRUPTION_TIMEOUT_SECONDS = 3

ACTIVITY_THRESHOLDS = {
    "low": [0, 33],
    "medium": [34, 66],
    "high": [67, 100],
}

VALIDATION_MANIFEST_PATH = "validation/clips.local.yml"
LOCAL_TRUSTED_NETWORK_ONLY = True

# Stream / WebRTC
SOURCE_ID = "drone_live"
MEDIAMTX_HOST = "127.0.0.1"
MEDIAMTX_WEBRTC_PORT = 8889
WEBRTC_PATH_TEMPLATE = "/{source_id}/whep"

# Overlay / WebSocket
STALE_OVERLAY_MS = 500
OVERLAY_HUB_QUEUE_MAXSIZE = 1

# Viewer / session lifecycle
VIEWER_HEARTBEAT_TIMEOUT_SECONDS = 30
IDLE_SHUTDOWN_DELAY_SECONDS = 10

# Backend defaults
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_DIST_DIR = "web/frontend/dist"
