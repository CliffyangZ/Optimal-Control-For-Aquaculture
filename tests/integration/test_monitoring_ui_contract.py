"""Monitoring contract tests for the current YOLO detection web flow."""

import pytest
from httpx import ASGITransport, AsyncClient

from config.setting import SOURCE_ID
from web.app import app
from web.schemas import DetectionPayload, DetectionsMessage
from core.models import DetectionHealthStatus, StreamStatus


@pytest.mark.asyncio
async def test_live_video_config_available_when_recognition_unavailable() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/config")

    assert response.status_code == 200
    data = response.json()
    assert data["webrtc_url"].endswith(f"/{SOURCE_ID}/whep")
    assert data["overlay_ws_url"].startswith("ws://")
    assert "ripple" not in data


def test_active_empty_detection_payload_clears_boxes() -> None:
    msg = DetectionsMessage(
        schema="detections.v1",
        source_id="drone_live",
        frame_id=7,
        timestamp_ms=1000,
        source_width=1280,
        source_height=720,
        stream_status=StreamStatus.LIVE,
        detection_status=DetectionHealthStatus.ACTIVE,
        detections=[],
    )

    assert msg.to_ws_dict()["detections"] == []


def test_detection_payload_contains_raw_labels_and_confidence() -> None:
    payload = DetectionPayload(class_name="lumpfish", confidence=0.93, x=0.1, y=0.2, width=0.3, height=0.4)

    assert payload.class_name == "lumpfish"
    assert payload.confidence == pytest.approx(0.93)
