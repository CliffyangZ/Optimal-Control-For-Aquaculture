"""WebSocket overlay and status payload contract tests."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from web.app import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def test_overlay_v1_schema_fields():
    """Verify overlay.v1 schema has all required fields."""
    from web.schemas import OverlayMessage, RegionPayload, ActivityPayload
    from core.models import StreamStatus, DetectionHealthStatus

    msg = OverlayMessage(
        schema="overlay.v1",
        source_id="drone_live",
        frame_id=1,
        timestamp_ms=1000,
        source_width=1280,
        source_height=720,
        stream_status=StreamStatus.LIVE,
        detection_status=DetectionHealthStatus.OK,
        activity=ActivityPayload(level="Medium", score=52, confidence=0.84),
        regions=[RegionPayload(id="r1", x=0.1, y=0.1, width=0.2, height=0.2)],
    )
    d = msg.to_ws_dict()
    assert d["schema"] == "overlay.v1"
    assert d["type"] == "overlay"
    assert "source_id" in d
    assert "frame_id" in d
    assert "timestamp_ms" in d
    assert "source_width" in d
    assert "source_height" in d
    assert d["stream_status"] == "live"
    assert d["detection_status"] == "ok"
    assert "regions" in d
    assert "activity" in d


def test_overlay_normalized_region_bounds():
    from web.schemas import RegionPayload
    import pytest

    r = RegionPayload(id="r1", x=0.0, y=0.0, width=1.0, height=1.0)
    assert r.x >= 0.0 and r.x <= 1.0
    assert r.y >= 0.0 and r.y <= 1.0
    assert r.width >= 0.0 and r.width <= 1.0
    assert r.height >= 0.0 and r.height <= 1.0

    with pytest.raises(Exception):
        RegionPayload(id="r2", x=1.5, y=0.0, width=0.2, height=0.2)


def test_status_v1_schema_fields():
    from web.schemas import StatusMessage
    from core.models import StreamStatus, DetectionHealthStatus

    msg = StatusMessage(
        schema="status.v1",
        source_id="drone_live",
        timestamp_ms=2000,
        stream_status=StreamStatus.LIVE,
        detection_status=DetectionHealthStatus.DELAYED,
        active_viewer_count=2,
        message="Detection delayed; live video remains active.",
    )
    d = msg.to_ws_dict()
    assert d["schema"] == "status.v1"
    assert d["type"] == "status"
    assert d["stream_status"] == "live"
    assert d["detection_status"] == "delayed"
    assert "active_viewer_count" in d


def test_overlay_message_source_dimensions_positive():
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
        detection_status=DetectionHealthStatus.OK,
    )
    d = msg.to_ws_dict()
    assert d["source_width"] > 0
    assert d["source_height"] > 0
