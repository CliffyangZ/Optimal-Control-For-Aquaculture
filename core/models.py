"""Shared dataclasses and enums for local CV monitoring."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import numpy as np

from .activity_levels import ActivityLevel


class StreamHealthState(str, Enum):
    """Stream health status during monitoring."""
    WAITING = "Waiting"
    ACTIVE = "Active"
    INTERRUPTED = "Interrupted"
    DETECTION_UNAVAILABLE = "Detection Unavailable"


class DetectionStatus(str, Enum):
    """Ripple detection status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


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
    """Continuous operator-controlled observation period."""
    session_id: str
    started_at: datetime
    stopped_at: Optional[datetime] = None
    health_state: StreamHealthState = StreamHealthState.WAITING
    view_mode: str = "processed"
    latest_frame: Optional[CurrentFrame] = None
    latest_detection: Optional[object] = None
