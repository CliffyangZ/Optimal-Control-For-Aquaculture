"""Shared test configuration and fixtures."""

import asyncio
import tempfile
from pathlib import Path

import pytest
from httpx import AsyncClient

from pi_controller.config import Config
from pi_controller.drivers.simulated import SimulatedFeederDriver
from pi_controller.services.controller_service import ControllerService
from pi_controller.services.realtime_hub import RealtimeHub
from pi_controller.storage.sqlite_store import SQLiteStore
from pi_controller.app import app


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def test_store(temp_db_path: Path) -> SQLiteStore:
    """Create a test SQLiteStore with temporary database."""
    return SQLiteStore(temp_db_path)


@pytest.fixture
def simulated_driver() -> SimulatedFeederDriver:
    """Create a simulated feeder driver."""
    return SimulatedFeederDriver()


@pytest.fixture
def realtime_hub() -> RealtimeHub:
    """Create a realtime hub."""
    return RealtimeHub()


@pytest.fixture
def controller_service(
    test_store: SQLiteStore,
    simulated_driver: SimulatedFeederDriver,
    realtime_hub: RealtimeHub,
) -> ControllerService:
    """Create a test controller service."""
    return ControllerService(
        store=test_store,
        feeder_driver=simulated_driver,
        realtime_hub=realtime_hub,
    )


@pytest.fixture
async def async_client(monkeypatch) -> AsyncClient:
    """Create an async HTTP client for FastAPI testing."""
    # Monkeypatch the config to use simulated driver and temp storage
    monkeypatch.setenv("PI_CONTROLLER_DRIVER_MODE", "simulated")
    monkeypatch.setenv("PI_CONTROLLER_STORAGE_DIR", tempfile.gettempdir())

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
