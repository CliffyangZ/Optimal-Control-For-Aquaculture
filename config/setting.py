"""Runtime configuration constants for aquaculture monitoring."""

import os
from pathlib import Path
from urllib.parse import urlparse


def _load_dotenv(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _stream_source_id(stream_url: str, default: str) -> str:
    parsed = urlparse(stream_url)
    path = parsed.path.strip("/")
    return path or default


def _stream_host(stream_url: str, default: str) -> str:
    return urlparse(stream_url).hostname or default


_load_dotenv()

# Stream Source
DEFAULT_STREAM_URL = "rtsp://192.168.137.157:8554/cam"
STREAM_URL = os.environ.get("STREAM_URL", DEFAULT_STREAM_URL)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_DISPLAY_FPS = 30
TARGET_DETECTION_FPS = 10
INTERRUPTION_TIMEOUT_SECONDS = 3

VALIDATION_MANIFEST_PATH = "validation/clips.local.yml"
LOCAL_TRUSTED_NETWORK_ONLY = True

# Stream / WebRTC
SOURCE_ID = os.environ.get("SOURCE_ID", _stream_source_id(STREAM_URL, "cam"))
MEDIAMTX_HOST = os.environ.get("MEDIAMTX_HOST", _stream_host(STREAM_URL, "192.168.137.157"))
MEDIAMTX_WEBRTC_PORT = _env_int("MEDIAMTX_WEBRTC_PORT", 8889)
WEBRTC_PATH_TEMPLATE = "/{source_id}/whep"

# Overlay / WebSocket
STALE_OVERLAY_MS = 500
OVERLAY_HUB_QUEUE_MAXSIZE = 1

# Approved YOLO web recognition
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "")
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "cpu")
YOLO_SUPPORTED_CLASSES = ("fish", "lumpfish", "feed", "debris")
YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.25"))

# Viewer / session lifecycle
VIEWER_HEARTBEAT_TIMEOUT_SECONDS = 30
IDLE_SHUTDOWN_DELAY_SECONDS = 10

# Backend defaults
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_DIST_DIR = "web/frontend/dist"
