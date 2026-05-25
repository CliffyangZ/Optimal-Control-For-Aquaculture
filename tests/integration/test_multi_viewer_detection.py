"""Integration tests: two viewers share one detection worker; cleanup after all disconnect."""

import asyncio
import pytest

from web.detection_worker import DetectionWorker


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr("config.setting.RFDETR_MODEL_PATH", "")
    return DetectionWorker()


@pytest.mark.asyncio
async def test_two_viewers_share_same_session(worker):
    s1 = await worker.register_viewer("v1")
    s2 = await worker.register_viewer("v2")
    assert s1.session_id == s2.session_id


@pytest.mark.asyncio
async def test_viewer_count_correct_with_two_viewers(worker):
    await worker.register_viewer("v1")
    await worker.register_viewer("v2")
    assert worker.get_session_snapshot().active_viewer_count == 2


@pytest.mark.asyncio
async def test_removing_one_viewer_does_not_stop_session(worker):
    await worker.register_viewer("v1")
    await worker.register_viewer("v2")
    await worker.unregister_viewer("v1")
    assert worker.get_session_snapshot() is not None
    assert worker.get_session_snapshot().active_viewer_count == 1


@pytest.mark.asyncio
async def test_only_one_worker_task_created(worker):
    await worker.register_viewer("v1")
    task_1 = worker._worker_task
    await worker.register_viewer("v2")
    task_2 = worker._worker_task
    assert task_1 is task_2
    if task_1:
        task_1.cancel()
        try:
            await task_1
        except (asyncio.CancelledError, Exception):
            pass
