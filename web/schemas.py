"""FastAPI response and WebSocket payload schemas."""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

from core.models import StreamStatus, DetectionHealthStatus


class RuntimeConfigResponse(BaseModel):
    source_id: str
    webrtc_url: str
    overlay_ws_url: str
    stale_overlay_ms: int = 500
    local_trusted_network_only: bool = True
    primary_browser_support: str = "Chrome and Edge are primary validation targets."


class SessionStatusResponse(BaseModel):
    source_id: str
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    active_viewer_count: int = Field(ge=0)
    started_at: Optional[str] = None
    last_frame_at: Optional[str] = None
    last_overlay_at: Optional[str] = None
    detection_fps: float = 0.0
    message: Optional[str] = None


class StopSessionRequest(BaseModel):
    viewer_id: Optional[str] = None


# --- WebSocket client message schemas ---

class ViewerHelloMessage(BaseModel):
    type: str = "viewer.hello"
    viewer_id: str
    overlay_enabled: bool = True


class ViewerHeartbeatMessage(BaseModel):
    type: str = "viewer.heartbeat"
    viewer_id: str
    timestamp_ms: int


class ViewerStopMessage(BaseModel):
    type: str = "viewer.stop"
    viewer_id: str


# --- WebSocket server message schemas ---


class DetectionPayload(BaseModel):
    class_name: Literal["fish", "feed", "debris", "lumpfish"]
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


class DetectorResult(BaseModel):
    """Results from a single detector (YOLO or RF-DETR)."""

    model_name: str = Field(description="Detector model name (yolo, rfdetr)")
    inference_ms: float = Field(ge=0.0)
    detections: list[DetectionPayload] = Field(default_factory=list)


class FeedAreaData(BaseModel):
    """Feed area analysis data for the current frame."""

    raw_area: float = Field(ge=0.0, description="Raw feed area in pixels")
    kalman_area: float = Field(ge=0.0, description="Kalman-filtered feed area")
    detection_count: int = Field(ge=0, description="Number of feed detections")


class FeedAnalysisHistory(BaseModel):
    """Historical feed area analysis for plotting."""

    frame_ids: list[int] = Field(default_factory=list)
    raw_areas: list[float] = Field(default_factory=list)
    kalman_areas: list[float] = Field(default_factory=list)


class DetectionsMessage(BaseModel):
    schema_: str = Field("detections.v2", alias="schema")
    type: str = "detections"
    source_id: str
    frame_id: int
    timestamp_ms: int
    source_width: int
    source_height: int
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    detections: list[DetectionPayload] = Field(default_factory=list)
    detectors: list[DetectorResult] = Field(default_factory=list, description="Results from multiple detectors")
    feed_area: Optional[FeedAreaData] = None
    feed_analysis: Optional[FeedAnalysisHistory] = None
    message: Optional[str] = None

    model_config = {"populate_by_name": True}

    def to_ws_dict(self) -> dict:
        d = self.model_dump(by_alias=True)
        d["stream_status"] = self.stream_status.value
        d["detection_status"] = self.detection_status.value
        return d


class StatusMessage(BaseModel):
    schema_: str = Field("status.v1", alias="schema")
    type: str = "status"
    source_id: str
    timestamp_ms: int
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    active_viewer_count: int
    message: Optional[str] = None

    model_config = {"populate_by_name": True}

    def to_ws_dict(self) -> dict:
        d = self.model_dump(by_alias=True)
        d["stream_status"] = self.stream_status.value
        d["detection_status"] = self.detection_status.value
        return d
