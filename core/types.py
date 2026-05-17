"""Shared dataclasses and enums for the YOLO26 detection pipeline.

Used by both the training pipeline (`core.train`) and the inference + feeding
control modules (`core.infer`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


CLASS_NAMES: tuple[str, ...] = ("fish", "feed", "debris", "lumpfish")
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Detection:
    """One bounding-box detection within a single frame.

    Bbox fields are normalized to [0.0, 1.0] relative to the source frame.
    """

    class_name: str
    class_index: int
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float


@dataclass
class DetectionResult:
    """All detections produced by YOLO26 for a single frame."""

    frame_id: int
    captured_at: datetime
    source_width: int
    source_height: int
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0


class BMduinoCommand(str, Enum):
    """Commands emitted by the FeedingStateMachine to the BMduino relay."""

    FEED = "FEED"
    STOP = "STOP"
    ALERT = "ALERT"


class FeedingState(str, Enum):
    """States of the threshold-based feeding control state machine."""

    IDLE = "IDLE"
    PRECHECK = "PRECHECK"
    DISPENSE = "DISPENSE"
    WAITING = "WAITING"
    CHECKING = "CHECKING"
    DONE = "DONE"
    ALERT = "ALERT"
