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

class ActivityPayload(BaseModel):
    level: str
    score: int
    confidence: float


class RegionPayload(BaseModel):
    id: str
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


class OverlayMessage(BaseModel):
    schema_: str = Field("overlay.v1", alias="schema")
    type: str = "overlay"
    source_id: str
    frame_id: int
    timestamp_ms: int
    source_width: int
    source_height: int
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    activity: Optional[ActivityPayload] = None
    regions: list[RegionPayload] = Field(default_factory=list)
    message: Optional[str] = None

    model_config = {"populate_by_name": True}

    def to_ws_dict(self) -> dict:
        d = self.model_dump(by_alias=True)
        d["stream_status"] = self.stream_status.value
        d["detection_status"] = self.detection_status.value
        return d


class DetectionPayload(BaseModel):
    class_name: Literal["fish", "feed", "debris", "lumpfish"]
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


class DetectionsMessage(BaseModel):
    schema_: str = Field("detections.v1", alias="schema")
    type: str = "detections"
    source_id: str
    frame_id: int
    timestamp_ms: int
    source_width: int
    source_height: int
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    detections: list[DetectionPayload] = Field(default_factory=list)

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
