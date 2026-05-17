"""Shared dataclasses and enums for the live monitoring web/streaming layer.

These are the cross-module DTOs consumed by `streaming/`, `web/`, and the
FastAPI overlay/WebSocket schemas. They are intentionally decoupled from the
spatiotemporal ripple algorithm in `core/ripple_algo.py` so that the algorithm
can evolve independently from the streaming surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Stream / detection health enums
# ---------------------------------------------------------------------------


class StreamStatus(str, Enum):
    """High-level state of the upstream RTSP/WebRTC video pipeline."""

    WAITING = "waiting"
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    STREAM_UNAVAILABLE = "stream_unavailable"
    DEGRADED = "degraded"


class DetectionHealthStatus(str, Enum):
    """Health of the detection pipeline as perceived by viewers."""

    OK = "ok"
    DELAYED = "delayed"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class DetectionStatus(str, Enum):
    """Per-frame availability of a detection result."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ViewerState(str, Enum):
    CONNECTING = "connecting"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    DISCONNECTED = "disconnected"
    TIMED_OUT = "timed_out"


# ---------------------------------------------------------------------------
# Frame DTOs
# ---------------------------------------------------------------------------


@dataclass
class CurrentFrame:
    """Latest raw frame produced by `LatestFrameReader`."""

    image: np.ndarray
    captured_at: datetime
    sequence_number: int
    source_width: int
    source_height: int


@dataclass
class VideoFrame:
    """Frame metadata as consumed by the detection worker."""

    frame_id: int
    captured_at: datetime
    source_width: int
    source_height: int
    image: np.ndarray

    def __post_init__(self) -> None:
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source_width and source_height must be positive")


# ---------------------------------------------------------------------------
# Overlay payload DTOs (server-side, before WS serialisation)
# ---------------------------------------------------------------------------


@dataclass
class ActivityRegion:
    """Normalized activity region in [0.0, 1.0] coordinates."""

    id: str
    x: float
    y: float
    width: float
    height: float


@dataclass
class ActivityBadge:
    """Headline activity summary for the current frame."""

    level: str
    score: int
    confidence: float
    # Optional fine-grained signals exposed by the spatiotemporal pipeline.
    # Defaults keep older callers backward-compatible.
    energy: float = 0.0
    energy_raw: float = 0.0


@dataclass
class DetectionOverlayPayload:
    """Overlay payload broadcast to viewers via WebSocket (server-side shape)."""

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


# ---------------------------------------------------------------------------
# Session / viewer DTOs
# ---------------------------------------------------------------------------


@dataclass
class ViewerSession:
    """A single connected browser viewer."""

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

    def __post_init__(self) -> None:
        if self.active_viewer_count < 0:
            raise ValueError("active_viewer_count must be >= 0")
