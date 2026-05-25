"""Contract tests for the detections.v1 WebSocket schema.

Validates that `DetectionsMessage` and `DetectionPayload` Pydantic models
enforce the field constraints specified in contracts/detections-v1.md.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models import DetectionHealthStatus, StreamStatus
from web.schemas import DetectionPayload, DetectionsMessage


def _valid_message(**overrides) -> dict:
    base = dict(
        source_id="drone_live",
        frame_id=1,
        timestamp_ms=1747459200000,
        source_width=1280,
        source_height=720,
        stream_status=StreamStatus.LIVE,
        detection_status=DetectionHealthStatus.ACTIVE,
        detections=[],
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# DetectionPayload field constraints
# ---------------------------------------------------------------------------

def test_detection_payload_valid_class_names() -> None:
    for name in ("fish", "feed", "debris", "lumpfish"):
        p = DetectionPayload(class_name=name, confidence=0.9, x=0.1, y=0.1, width=0.1, height=0.1)
        assert p.class_name == name


def test_detection_payload_accepts_model_returned_class_names() -> None:
    payload = DetectionPayload(class_name="shrimp", class_index=9, confidence=0.9, x=0.1, y=0.1, width=0.1, height=0.1)

    assert payload.class_name == "shrimp"
    assert payload.class_index == 9


def test_detection_payload_bbox_clamped_to_zero_one() -> None:
    # Values at boundary should be accepted
    p = DetectionPayload(class_name="fish", confidence=1.0, x=0.0, y=0.0, width=1.0, height=1.0)
    assert p.x == 0.0
    assert p.width == 1.0


def test_detection_payload_rejects_negative_x() -> None:
    with pytest.raises(ValidationError):
        DetectionPayload(class_name="fish", confidence=0.9, x=-0.1, y=0.1, width=0.1, height=0.1)


def test_detection_payload_rejects_x_over_one() -> None:
    with pytest.raises(ValidationError):
        DetectionPayload(class_name="fish", confidence=0.9, x=1.1, y=0.1, width=0.1, height=0.1)


def test_detection_payload_rejects_confidence_over_one() -> None:
    with pytest.raises(ValidationError):
        DetectionPayload(class_name="fish", confidence=1.1, x=0.1, y=0.1, width=0.1, height=0.1)


def test_detection_payload_rejects_negative_confidence() -> None:
    with pytest.raises(ValidationError):
        DetectionPayload(class_name="fish", confidence=-0.1, x=0.1, y=0.1, width=0.1, height=0.1)


# ---------------------------------------------------------------------------
# DetectionsMessage top-level fields
# ---------------------------------------------------------------------------

def test_detections_message_schema_field_is_exact_string() -> None:
    msg = DetectionsMessage(**_valid_message())
    ws_dict = msg.to_ws_dict()
    assert ws_dict["schema"] == "detections.v1"


def test_detections_message_type_field() -> None:
    msg = DetectionsMessage(**_valid_message())
    assert msg.type == "detections"


def test_detections_message_empty_detections_is_valid() -> None:
    msg = DetectionsMessage(**_valid_message(detections=[]))
    ws = msg.to_ws_dict()
    assert ws["detections"] == []


def test_detections_message_with_two_detections() -> None:
    dets = [
        DetectionPayload(class_name="fish", confidence=0.921, x=0.312, y=0.451, width=0.087, height=0.094),
        DetectionPayload(class_name="feed", confidence=0.873, x=0.654, y=0.213, width=0.043, height=0.039),
    ]
    msg = DetectionsMessage(**_valid_message(detections=dets))
    ws = msg.to_ws_dict()
    assert len(ws["detections"]) == 2
    assert ws["detections"][0]["class_name"] == "fish"
    assert ws["detections"][1]["class_name"] == "feed"


def test_detections_message_stream_status_serialized_as_value() -> None:
    msg = DetectionsMessage(**_valid_message(stream_status=StreamStatus.LIVE))
    ws = msg.to_ws_dict()
    assert ws["stream_status"] == "live"


def test_detections_message_detection_status_serialized_as_value() -> None:
    msg = DetectionsMessage(**_valid_message(detection_status=DetectionHealthStatus.ACTIVE))
    ws = msg.to_ws_dict()
    assert ws["detection_status"] == "active"


def test_detections_message_detection_status_unavailable() -> None:
    msg = DetectionsMessage(**_valid_message(detection_status=DetectionHealthStatus.UNAVAILABLE))
    ws = msg.to_ws_dict()
    assert ws["detection_status"] == "unavailable"


def test_detections_message_accepts_active_detection_statuses() -> None:
    expected = {
        DetectionHealthStatus.LOADING: "loading",
        DetectionHealthStatus.ACTIVE: "active",
        DetectionHealthStatus.STALE: "stale",
        DetectionHealthStatus.UNAVAILABLE: "unavailable",
    }

    for status, serialized in expected.items():
        msg = DetectionsMessage(**_valid_message(detection_status=status))
        assert msg.to_ws_dict()["detection_status"] == serialized


# ---------------------------------------------------------------------------
# to_ws_dict round-trip
# ---------------------------------------------------------------------------

def test_to_ws_dict_contains_required_top_level_keys() -> None:
    msg = DetectionsMessage(**_valid_message())
    ws = msg.to_ws_dict()
    required = {"schema", "type", "source_id", "frame_id", "timestamp_ms",
                "source_width", "source_height", "stream_status", "detection_status", "detections"}
    assert required.issubset(ws.keys())


def test_to_ws_dict_detection_payload_keys() -> None:
    det = DetectionPayload(class_name="debris", confidence=0.7, x=0.2, y=0.3, width=0.05, height=0.06)
    msg = DetectionsMessage(**_valid_message(detections=[det]))
    ws = msg.to_ws_dict()
    d = ws["detections"][0]
    for key in ("class_name", "class_index", "confidence", "x", "y", "width", "height"):
        assert key in d, f"Missing key '{key}' in detection payload"
