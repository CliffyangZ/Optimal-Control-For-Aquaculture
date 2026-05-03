"""Shared dataclasses and enums for ripple monitoring."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import numpy as np

from .activity_levels import ActivityLevel


class StreamHealthState(str, Enum):
    """Stream health status during monitoring (legacy Streamlit UI)."""
    WAITING = "Waiting"
    ACTIVE = "Active"
    INTERRUPTED = "Interrupted"
    DETECTION_UNAVAILABLE = "Detection Unavailable"


class DetectionStatus(str, Enum):
    """Ripple detection status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


# --- FastAPI / WebRTC streaming enums ---

class StreamStatus(str, Enum):
    WAITING = "waiting"
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    STREAM_UNAVAILABLE = "stream_unavailable"
    DEGRADED = "degraded"


class DetectionHealthStatus(str, Enum):
    OK = "ok"
    DELAYED = "delayed"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ViewerState(str, Enum):
    CONNECTING = "connecting"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    DISCONNECTED = "disconnected"
    TIMED_OUT = "timed_out"


# --- Legacy Streamlit dataclasses ---

@dataclass
class CurrentFrame:
    """Latest frame available for display or analysis."""
    image: np.ndarray
    captured_at: datetime
    sequence_number: int
    source_width: int
    source_height: int


@dataclass
class RippleDetectionResult:
    """Most recent completed ripple detection analysis result."""
    status: DetectionStatus
    activity_score: Optional[int] = None
    activity_level: Optional[ActivityLevel] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    visual_marker: Optional[dict] = None
    frame_sequence_number: Optional[int] = None
    completed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.status == DetectionStatus.AVAILABLE:
            if self.activity_score is None or self.activity_level is None:
                raise ValueError("Available detection must have score and level")
        else:
            if self.activity_score is not None or self.activity_level is not None:
                raise ValueError("Unavailable detection must not have score or level")
            if self.reason is None:
                raise ValueError("Unavailable detection must have a reason")


@dataclass
class MonitoringSession:
    """Continuous operator-controlled observation period (legacy Streamlit UI)."""
    session_id: str
    started_at: datetime
    stopped_at: Optional[datetime] = None
    health_state: StreamHealthState = StreamHealthState.WAITING
    view_mode: str = "processed"
    latest_frame: Optional[CurrentFrame] = None
    latest_detection: Optional[RippleDetectionResult] = None


# --- FastAPI streaming domain models ---

@dataclass
class VideoFrame:
    """One frame read by backend detection."""
    frame_id: int
    captured_at: datetime
    source_width: int
    source_height: int
    image: np.ndarray

    def __post_init__(self):
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source_width and source_height must be positive")


@dataclass
class ActivityRegion:
    """Normalized activity region (0.0–1.0 coordinates)."""
    id: str
    x: float
    y: float
    width: float
    height: float


@dataclass
class ActivityBadge:
    level: str
    score: int
    confidence: float


@dataclass
class DetectionOverlayPayload:
    """Message sent to browser viewers over WebSocket."""
    schema: str
    source_id: str
    frame_id: int
    timestamp_ms: int
    source_width: int
    source_height: int
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    activity: Optional[ActivityBadge] = None
    regions: list[ActivityRegion] = field(default_factory=list)
    message: Optional[str] = None


@dataclass
class ViewerSession:
    """One connected browser viewer."""
    viewer_id: str
    connected_at: datetime
    last_heartbeat_at: datetime
    state: ViewerState = ViewerState.CONNECTING
    overlay_enabled: bool = True


@dataclass
class LiveStreamSession:
    """Shared monitoring lifecycle for the configured stream."""
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

    def __post_init__(self):
        if self.active_viewer_count < 0:
            raise ValueError("active_viewer_count must be >= 0")


@dataclass
class RuntimeConfig:
    """Client-facing configuration returned by the backend."""
    source_id: str
    webrtc_url: str
    overlay_ws_url: str
    stale_overlay_ms: int
    local_trusted_network_only: bool
    primary_browser_support: str = "Chrome and Edge are primary validation targets."
