"""FastAPI response and WebSocket payload schemas."""

from __future__ import annotations
from typing import Optional
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
    # Returned only by POST /session/start; null on all other endpoints.
    viewer_id: Optional[str] = None
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
    # Smoothed (EMA) and raw (per-frame) ripple energy in [0.0, 1.0]. The
    # `score` field above is a 0–100 integer projection of `energy` kept for
    # backward compatibility with the existing badge UI; the chart in the
    # frontend plots `energy` and `energy_raw` directly.
    energy: float = 0.0
    energy_raw: float = 0.0
    # Detector-internal frame index (monotonically increasing). Useful for
    # debugging chart gaps / frame drops on the client.
    frame_index: int = 0


class RegionPayload(BaseModel):
    id: str
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)
    # Candidate structure score in [0, 1] — drives ring size / opacity on the
    # frontend energy-centre overlay.
    score: float = Field(default=0.0, ge=0.0, le=1.0)


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
