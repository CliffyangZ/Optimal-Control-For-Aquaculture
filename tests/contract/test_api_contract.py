"""Contract tests for /config, /session/start, /session/stop, and /status."""

from urllib.parse import urlparse

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from config.setting import MEDIAMTX_HOST
from web.app import app
from web.app import LOOPBACK_MEDIAMTX_HOSTS


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# --- /config ---

@pytest.mark.asyncio
async def test_config_returns_required_fields(client):
    r = await client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "source_id" in body
    assert "webrtc_url" in body
    assert "overlay_ws_url" in body
    assert "stale_overlay_ms" in body
    assert body["local_trusted_network_only"] is True


@pytest.mark.asyncio
async def test_config_webrtc_url_contains_source_id(client):
    r = await client.get("/config")
    body = r.json()
    assert body["source_id"] in body["webrtc_url"]


@pytest.mark.asyncio
async def test_config_webrtc_url_uses_browser_visible_host_for_loopback_mediamtx(client):
    r = await client.get("/config", headers={"host": "192.168.10.12:8000"})
    body = r.json()
    expected_host = "192.168.10.12" if MEDIAMTX_HOST in LOOPBACK_MEDIAMTX_HOSTS else MEDIAMTX_HOST
    assert urlparse(body["webrtc_url"]).hostname == expected_host


@pytest.mark.asyncio
async def test_config_stale_overlay_ms_positive(client):
    r = await client.get("/config")
    assert r.json()["stale_overlay_ms"] > 0


@pytest.mark.asyncio
async def test_config_overlay_ws_url_starts_with_ws(client):
    r = await client.get("/config")
    url = r.json()["overlay_ws_url"]
    assert url.startswith("ws://") or url.startswith("wss://")


# --- /status ---

@pytest.mark.asyncio
async def test_status_returns_required_fields(client):
    r = await client.get("/status")
    assert r.status_code == 200
    body = r.json()
    for field in ("source_id", "stream_status", "detection_status", "active_viewer_count"):
        assert field in body


@pytest.mark.asyncio
async def test_status_stream_status_is_valid_enum(client):
    r = await client.get("/status")
    valid = {"waiting", "connecting", "live", "reconnecting", "stream_unavailable", "degraded"}
    assert r.json()["stream_status"] in valid


@pytest.mark.asyncio
async def test_status_detection_status_is_valid_enum(client):
    r = await client.get("/status")
    valid = {"ok", "delayed", "degraded", "unavailable"}
    assert r.json()["detection_status"] in valid


@pytest.mark.asyncio
async def test_status_active_viewer_count_non_negative(client):
    r = await client.get("/status")
    assert r.json()["active_viewer_count"] >= 0


# --- /session/start ---

@pytest.mark.asyncio
async def test_session_start_returns_session_status(client):
    r = await client.post("/session/start")
    assert r.status_code == 200
    body = r.json()
    for field in ("source_id", "stream_status", "detection_status", "active_viewer_count"):
        assert field in body


@pytest.mark.asyncio
async def test_session_start_increments_viewer_count(client):
    before = (await client.get("/status")).json()["active_viewer_count"]
    await client.post("/session/start")
    after = (await client.get("/status")).json()["active_viewer_count"]
    assert after > before


# --- /session/stop ---

@pytest.mark.asyncio
async def test_session_stop_returns_session_status(client):
    r = await client.post("/session/stop", json={})
    assert r.status_code == 200
    body = r.json()
    for field in ("source_id", "stream_status", "detection_status", "active_viewer_count"):
        assert field in body


@pytest.mark.asyncio
async def test_session_stop_with_viewer_id(client):
    start = await client.post("/session/start")
    msg = start.json().get("message", "")
    viewer_id = msg.split("viewer_id:")[1] if "viewer_id:" in msg else None
    r = await client.post("/session/stop", json={"viewer_id": viewer_id})
    assert r.status_code == 200
