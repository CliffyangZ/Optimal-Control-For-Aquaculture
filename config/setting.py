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
