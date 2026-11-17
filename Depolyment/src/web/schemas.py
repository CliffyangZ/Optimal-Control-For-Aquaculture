"""Pydantic schemas for WebSocket messages and HTTP responses.

Defines the wire format for:
- detections.v1 - real-time detection overlays
- analytics.v1 - hunger level and feed area analysis
- status.v1 - stream and detection health status
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from core.models import DetectionHealthStatus, StreamStatus


class DetectionPayload(BaseModel):
    """One bounding-box detection in normalized coordinates [0.0, 1.0]."""

    class_name: str
    class_index: Optional[int] = None
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @field_validator("confidence", "x", "y", "width", "height")
    @classmethod
    def validate_ranges(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Value {v} not in range [0.0, 1.0]")
        return v


class DetectionsMessage(BaseModel):
    """WebSocket message for real-time detection overlays (schema: detections.v1)."""

    schema: str = Field(default="detections.v1", frozen=True)
    type: str = Field(default="detections", frozen=True)
    source_id: str
    frame_id: int
    timestamp_ms: int
    source_width: int
    source_height: int
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    detections: list[DetectionPayload]

    def to_ws_dict(self) -> dict:
        """Serialize to WebSocket dictionary with enum values."""
        detections_list = []
        for det in self.detections:
            detections_list.append(det.model_dump())

        return {
            "schema": self.schema,
            "type": self.type,
            "source_id": self.source_id,
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "stream_status": self.stream_status.value,
            "detection_status": self.detection_status.value,
            "detections": detections_list,
        }


class StatusMessage(BaseModel):
    """WebSocket message for stream/detection status (schema: status.v1)."""

    schema: str = Field(default="status.v1", frozen=True)
    type: str = Field(default="status", frozen=True)
    source_id: str
    timestamp_ms: int
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    active_viewer_count: int
    message: str

    def to_ws_dict(self) -> dict:
        """Serialize to WebSocket dictionary with enum values."""
        return {
            "schema": self.schema,
            "type": self.type,
            "source_id": self.source_id,
            "timestamp_ms": self.timestamp_ms,
            "stream_status": self.stream_status.value,
            "detection_status": self.detection_status.value,
            "active_viewer_count": self.active_viewer_count,
            "message": self.message,
        }


class AnalyticsMessage(BaseModel):
    """WebSocket message for feed area analysis and hunger level (schema: analytics.v1)."""

    schema: str = Field(default="analytics.v1", frozen=True)
    type: str = Field(default="analytics", frozen=True)
    source_id: str
    timestamp_ms: int
    hunger_level: str
    consumption_rate: float
    frame_ids: list[int]
    raw_area: list[float]
    kalman_area: list[float]
    ema_area: list[float]

    def to_ws_dict(self) -> dict:
        """Serialize to WebSocket dictionary."""
        return {
            "schema": self.schema,
            "type": self.type,
            "source_id": self.source_id,
            "timestamp_ms": self.timestamp_ms,
            "hunger_level": self.hunger_level,
            "consumption_rate": self.consumption_rate,
            "frame_ids": self.frame_ids,
            "raw_area": self.raw_area,
            "kalman_area": self.kalman_area,
            "ema_area": self.ema_area,
        }


class ConfigResponse(BaseModel):
    """Response from GET /config."""

    source_id: str
    webrtc_url: str
    hls_url: str
    overlay_ws_url: str
    stale_overlay_ms: int
    local_trusted_network_only: bool


class SessionStatusResponse(BaseModel):
    """Response from GET /status or POST /session/start|stop."""

    source_id: str
    stream_status: str
    detection_status: str
    active_viewer_count: int
