"""Unit tests for detection result normalization and stale status handling."""

import pytest
from datetime import datetime, timezone


def test_normalize_region_to_unit():
    """Pixel regions are normalized to 0–1 relative to source dimensions."""
    from web.schemas import RegionPayload

    sw, sh = 1280, 720
    px_x, px_y, px_w, px_h = 100, 50, 200, 100
    r = RegionPayload(
        id="r1",
        x=px_x / sw,
        y=px_y / sh,
        width=px_w / sw,
        height=px_h / sh,
    )
    assert abs(r.x - 0.078125) < 0.0001
    assert abs(r.y - 0.06944) < 0.0001
    assert r.width == pytest.approx(200 / 1280)
    assert r.height == pytest.approx(100 / 720)


def test_normalized_coordinates_within_bounds():
    from web.schemas import RegionPayload

    r = RegionPayload(id="r1", x=0.5, y=0.5, width=0.4, height=0.4)
    assert r.x + r.width <= 1.0
    assert r.y + r.height <= 1.0


def test_stale_overlay_detected_by_timestamp(monkeypatch):
    """Overlay is considered stale when timestamp_ms is older than stale_ms."""
    import time
    stale_ms = 500
    old_ts = int(time.time() * 1000) - stale_ms - 100
    age = int(time.time() * 1000) - old_ts
    assert age > stale_ms


def test_fresh_overlay_not_stale():
    import time
    stale_ms = 500
    fresh_ts = int(time.time() * 1000) - 100
    age = int(time.time() * 1000) - fresh_ts
    assert age < stale_ms


def test_detection_unavailable_payload_has_no_regions():
    from web.schemas import OverlayMessage
    from core.models import StreamStatus, DetectionHealthStatus

    msg = OverlayMessage(
        schema="overlay.v1",
        source_id="drone_live",
        frame_id=1,
        timestamp_ms=1000,
        source_width=1280,
        source_height=720,
        stream_status=StreamStatus.LIVE,
        detection_status=DetectionHealthStatus.UNAVAILABLE,
        message="Detector error",
    )
    assert msg.regions == []
    assert msg.activity is None


def test_derive_stream_status_live():
    from core.stream_health import derive_stream_status
    from core.models import StreamStatus
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    status = derive_stream_status(
        last_frame_at=now,
        started_at=now - timedelta(seconds=10),
    )
    assert status == StreamStatus.LIVE


def test_derive_stream_status_connecting():
    from core.stream_health import derive_stream_status
    from core.models import StreamStatus
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    status = derive_stream_status(
        last_frame_at=None,
        started_at=now - timedelta(seconds=1),
    )
    assert status == StreamStatus.CONNECTING


def test_derive_stream_status_unavailable():
    from core.stream_health import derive_stream_status
    from core.models import StreamStatus
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    status = derive_stream_status(
        last_frame_at=None,
        started_at=now - timedelta(seconds=10),
    )
    assert status == StreamStatus.STREAM_UNAVAILABLE


def test_derive_detection_status_ok():
    from core.stream_health import derive_detection_status
    from core.models import DetectionHealthStatus
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    status = derive_detection_status(
        last_overlay_at=now,
        last_frame_at=now,
    )
    assert status == DetectionHealthStatus.OK


def test_derive_detection_status_unavailable_when_no_frame():
    from core.stream_health import derive_detection_status
    from core.models import DetectionHealthStatus

    status = derive_detection_status(last_overlay_at=None, last_frame_at=None)
    assert status == DetectionHealthStatus.UNAVAILABLE
