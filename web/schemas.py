"""Pydantic schemas for the live RF-DETR monitoring API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from core.models import DetectionHealthStatus, StreamStatus


class ConfigResponse(BaseModel):
    source_id: str
    webrtc_url: str
    overlay_ws_url: str
    stale_overlay_ms: int
    local_trusted_network_only: bool
    detector_model_configured: bool
    confidence_threshold: float
    device: str


class SessionStatusResponse(BaseModel):
    source_id: str
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    active_viewer_count: int
    detection_fps: float = 0.0
    last_frame_at: str | None = None
    last_overlay_at: str | None = None
    message: str = ""
    viewer_id: str | None = None


class SessionStopRequest(BaseModel):
    viewer_id: str | None = None


class HeartbeatRequest(BaseModel):
    viewer_id: str


class DetectionPayload(BaseModel):
    class_name: str = Field(min_length=1)
    class_index: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def bbox_must_fit_unit_square(self) -> "DetectionPayload":
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("normalized detection bbox must fit within [0, 1]")
        return self


class DetectionsMessage(BaseModel):
    schema: Literal["detections.v1"] = "detections.v1"
    type: Literal["detections"] = "detections"
    source_id: str
    frame_id: int
    timestamp_ms: int
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    detections: list[DetectionPayload]
    inference_ms: float = 0.0
    detection_fps: float = 0.0
    message: str = ""

    def to_ws_dict(self) -> dict:
        return self.model_dump(mode="json")


class StatusMessage(BaseModel):
    schema: Literal["status.v1"] = "status.v1"
    type: Literal["status"] = "status"
    source_id: str
    timestamp_ms: int
    stream_status: StreamStatus
    detection_status: DetectionHealthStatus
    active_viewer_count: int = Field(ge=0)
    detection_fps: float = 0.0
    message: str = ""

    def to_ws_dict(self) -> dict:
        return self.model_dump(mode="json")
