"""Web-layer domain models for the FastAPI streaming service.

These models are shared between `web/detection_worker.py`, `web/schemas.py`,
and `web/app.py`. They are distinct from the ML pipeline types in `core/types.py`.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class CurrentFrame:
    """One raw frame from the RTSP stream (used by LatestFrameReader)."""
    image: np.ndarray
    captured_at: datetime
    sequence_number: int
    source_width: int
    source_height: int


class StreamStatus(str, Enum):
    WAITING = "waiting"
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    STREAM_UNAVAILABLE = "stream_unavailable"
    ERROR = "error"
    DEGRADED = "degraded"


class DetectionHealthStatus(str, Enum):
    LOADING = "loading"
    ACTIVE = "active"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class ViewerState(str, Enum):
    CONNECTING = "connecting"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    DISCONNECTED = "disconnected"
    TIMED_OUT = "timed_out"


@dataclass
class VideoFrame:
    frame_id: int
    captured_at: datetime
    source_width: int
    source_height: int
    image: np.ndarray

    def __post_init__(self) -> None:
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source_width and source_height must be positive")


@dataclass
class ViewerSession:
    viewer_id: str
    connected_at: datetime
    last_heartbeat_at: datetime
    state: ViewerState = ViewerState.CONNECTING
    overlay_enabled: bool = True


@dataclass
class LiveStreamSession:
    session_id: str
    source_id: str
    started_at: datetime
    stopped_at: Optional[datetime] = None
    stream_status: StreamStatus = StreamStatus.WAITING
    detection_status: DetectionHealthStatus = DetectionHealthStatus.UNAVAILABLE
    active_viewer_count: int = 0
    last_frame_at: Optional[datetime] = None
    last_overlay_at: Optional[datetime] = None
    detection_fps: float = 0.0

    def __post_init__(self) -> None:
        if self.active_viewer_count < 0:
            raise ValueError("active_viewer_count must be >= 0")
