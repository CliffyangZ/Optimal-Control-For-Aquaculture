"""Unit tests for detections.v1 payloads and active detection statuses."""

from datetime import datetime, timedelta, timezone

import pytest


def test_detection_payload_normalized_coordinates_within_bounds() -> None:
    from web.schemas import DetectionPayload

    detection = DetectionPayload(class_name="fish", confidence=0.9, x=0.5, y=0.5, width=0.4, height=0.4)

    assert detection.x + detection.width <= 1.0
    assert detection.y + detection.height <= 1.0


def test_detection_payload_rejects_out_of_bounds_coordinate() -> None:
    from web.schemas import DetectionPayload

    with pytest.raises(Exception):
        DetectionPayload(class_name="fish", confidence=0.9, x=1.5, y=0.0, width=0.2, height=0.2)


def test_derive_detection_status_active() -> None:
    from core.models import DetectionHealthStatus
    from core.stream_health import derive_detection_status

    now = datetime.now(timezone.utc)

    assert derive_detection_status(last_overlay_at=now, last_frame_at=now) == DetectionHealthStatus.ACTIVE


def test_derive_detection_status_loading_without_payload() -> None:
    from core.models import DetectionHealthStatus
    from core.stream_health import derive_detection_status

    now = datetime.now(timezone.utc)

    assert derive_detection_status(last_overlay_at=None, last_frame_at=now) == DetectionHealthStatus.LOADING


def test_derive_detection_status_stale() -> None:
    from core.models import DetectionHealthStatus
    from core.stream_health import derive_detection_status

    now = datetime.now(timezone.utc)

    assert derive_detection_status(
        last_overlay_at=now - timedelta(seconds=3),
        last_frame_at=now,
        stale_seconds=1,
    ) == DetectionHealthStatus.STALE


def test_derive_detection_status_unavailable_when_detector_unavailable() -> None:
    from core.models import DetectionHealthStatus
    from core.stream_health import derive_detection_status

    assert derive_detection_status(None, None, detector_available=False) == DetectionHealthStatus.UNAVAILABLE
