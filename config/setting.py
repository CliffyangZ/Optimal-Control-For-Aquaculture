"""Configuration constants for spatiotemporal water ripple analysis monitoring.

Includes stream source settings, spatial grid dimensions, feature fusion weights,
optical flow and edge detection parameters, activity level thresholds, and UI settings.
All weights must sum to 1.0 and threshold values must be in valid ranges [0, 1].
"""

# Stream Source
STREAM_URL = "rtsp://127.0.0.1:8554/drone_live"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_DISPLAY_FPS = 30
TARGET_DETECTION_FPS = 10
INTERRUPTION_TIMEOUT_SECONDS = 3

# Spatial Grid
GRID_ROWS = 8
GRID_COLS = 8

# Feature Weights (must sum to 1.0)
WEIGHT_SPATIAL = 0.25
WEIGHT_EXPANSION = 0.35
WEIGHT_MOTION = 0.25
WEIGHT_EDGE_CHANGE = 0.15

# Sliding Window
WINDOW_FRAMES = 30

# Optical Flow (Farneback parameters)
FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 21
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2

# Edge Detection (Canny)
CANNY_LOW = 50
CANNY_HIGH = 150
BLUR_KERNEL = (5, 5)

# Activity Level Thresholds
ACTIVITY_LOW_THRESHOLD = 0.2
ACTIVITY_HIGH_THRESHOLD = 0.5

# Temporal Feature Thresholds
MOTION_THRESHOLD = 12
FLOW_MAG_THRESHOLD = 0.2
FLOW_SAMPLE_STEP = 4
RIPPLE_R_MIN = 10
RIPPLE_R_MAX = 120

# UI Settings
HEATMAP_COLORMAP = "HOT"
HEATMAP_ALPHA = 0.5

# Legacy thresholds (for backward compatibility)
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
