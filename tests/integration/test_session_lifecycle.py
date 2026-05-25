"""Integration tests for single-viewer start, stop, and cleanup lifecycle."""

import asyncio
import pytest

from web.detection_worker import DetectionWorker


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr("config.setting.RFDETR_MODEL_PATH", "")
    return DetectionWorker()


@pytest.mark.asyncio
async def test_register_viewer_creates_session(worker):
    session = await worker.register_viewer("v1")
    assert session is not None
    assert session.source_id is not None
    assert session.active_viewer_count == 1


@pytest.mark.asyncio
async def test_register_multiple_viewers_shares_session(worker):
    s1 = await worker.register_viewer("v1")
    s2 = await worker.register_viewer("v2")
    assert s1.session_id == s2.session_id
    assert s2.active_viewer_count == 2


@pytest.mark.asyncio
async def test_unregister_viewer_decrements_count(worker):
    await worker.register_viewer("v1")
    await worker.register_viewer("v2")
    await worker.unregister_viewer("v1")
    assert worker.get_session_snapshot().active_viewer_count == 1


@pytest.mark.asyncio
async def test_session_snapshot_returns_none_before_first_viewer(worker):
    assert worker.get_session_snapshot() is None


@pytest.mark.asyncio
async def test_update_heartbeat_does_not_raise(worker):
    await worker.register_viewer("v1")
    await worker.update_heartbeat("v1")


@pytest.mark.asyncio
async def test_update_heartbeat_unknown_viewer_is_safe(worker):
    await worker.update_heartbeat("does-not-exist")


@pytest.mark.asyncio
async def test_unregister_unknown_viewer_is_safe(worker):
    result = await worker.unregister_viewer("ghost")
    assert result is None


@pytest.mark.asyncio
async def test_worker_task_started_on_first_viewer(worker):
    await worker.register_viewer("v1")
    assert worker._worker_task is not None
    worker._worker_task.cancel()
    try:
        await worker._worker_task
    except (asyncio.CancelledError, Exception):
        pass
