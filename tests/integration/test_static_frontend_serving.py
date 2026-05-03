"""Backend tests for FastAPI routes when static assets are not present."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from web.app import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_api_routes_available_without_frontend_build(client):
    """API routes work even when the frontend dist/ directory doesn't exist."""
    r = await client.get("/config")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_status_route_available(client):
    r = await client.get("/status")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_session_start_route_available(client):
    r = await client.post("/session/start")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_session_stop_route_available(client):
    r = await client.post("/session/stop", json={})
    assert r.status_code == 200
