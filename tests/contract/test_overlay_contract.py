"""Compatibility WebSocket path now carries detections.v1 and status.v1 messages."""

from core.models import DetectionHealthStatus, StreamStatus
from web.schemas import DetectionPayload, DetectionsMessage, StatusMessage


def test_detection_message_schema_fields() -> None:
    msg = DetectionsMessage(
        schema="detections.v1",
        source_id="drone_live",
        frame_id=1,
        timestamp_ms=1000,
        source_width=1280,
        source_height=720,
        stream_status=StreamStatus.LIVE,
        detection_status=DetectionHealthStatus.ACTIVE,
        detections=[DetectionPayload(class_name="fish", confidence=0.84, x=0.1, y=0.1, width=0.2, height=0.2)],
    )

    data = msg.to_ws_dict()

    assert data["schema"] == "detections.v1"
    assert data["type"] == "detections"
    assert data["stream_status"] == "live"
    assert data["detection_status"] == "active"
    assert data["detections"][0]["class_name"] == "fish"


def test_status_v1_uses_active_detection_statuses() -> None:
    msg = StatusMessage(
        schema="status.v1",
        source_id="drone_live",
        timestamp_ms=2000,
        stream_status=StreamStatus.LIVE,
        detection_status=DetectionHealthStatus.STALE,
        active_viewer_count=2,
        message="Detection data is stale",
    )

    data = msg.to_ws_dict()

    assert data["schema"] == "status.v1"
    assert data["type"] == "status"
    assert data["detection_status"] == "stale"
