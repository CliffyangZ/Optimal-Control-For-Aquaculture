# Contract: Configuration

Configuration is a Python-readable local file owned by `config/`. Values are loaded by streaming, detection, validation, and UI modules; core algorithms must not hard-code source URLs or thresholds.

## Required Settings

```python
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
```

## Rules

- `STREAM_URL` must be the local relay URL consumed by OpenCV, not a public endpoint.
- Display target defaults must remain 720p and 30 FPS for the MVP.
- Detection target defaults must remain at least 10 updates per second under normal local conditions.
- `INTERRUPTION_TIMEOUT_SECONDS` must be `3`.
- Activity thresholds must cover every integer score from 0 through 100 exactly once.
- `LOCAL_TRUSTED_NETWORK_ONLY` must remain true for MVP documentation and UI warnings.
- Video clips referenced by `VALIDATION_MANIFEST_PATH` must not be required in git.

## Consumers

- `streaming/latest_frame_reader.py` consumes `STREAM_URL`, target dimensions, and interruption timeout.
- `core/activity_levels.py` consumes `ACTIVITY_THRESHOLDS`.
- `web/app.py` consumes target rates and local-network warning settings.
- `validation/clip_manifest.py` consumes `VALIDATION_MANIFEST_PATH`.
