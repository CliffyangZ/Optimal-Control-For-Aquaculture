"""Unit tests for the Raspberry Pi schedule runner."""

from datetime import datetime

import pytest

from pi_controller.domain.models import FeedRunSource, FeedRunStatus, Weekday


WEEKDAYS = [
    Weekday.MON,
    Weekday.TUE,
    Weekday.WED,
    Weekday.THU,
    Weekday.FRI,
    Weekday.SAT,
    Weekday.SUN,
]


@pytest.mark.asyncio
async def test_due_schedule_starts_feed(controller_service):
    now = datetime(2026, 5, 22, 8, 30, 15)
    schedule = await controller_service.create_schedule(
        name="Morning feed",
        enabled=True,
        weekdays=[WEEKDAYS[now.weekday()]],
        times=["08:30"],
        duration_seconds=1,
    )

    runs = await controller_service.run_due_schedules(now)

    assert len(runs) == 1
    assert runs[0].source == FeedRunSource.SCHEDULE
    assert runs[0].schedule_id == schedule.id
    assert runs[0].planned_at == datetime(2026, 5, 22, 8, 30)
    assert controller_service.feeder_service.get_active_run() == runs[0]

    await controller_service.stop_feed()


@pytest.mark.asyncio
async def test_due_schedule_is_attempted_once_per_minute(controller_service):
    now = datetime(2026, 5, 22, 8, 30, 15)
    await controller_service.create_schedule(
        name="Morning feed",
        enabled=True,
        weekdays=[WEEKDAYS[now.weekday()]],
        times=["08:30"],
        duration_seconds=1,
    )

    first_runs = await controller_service.run_due_schedules(now)
    second_runs = await controller_service.run_due_schedules(now.replace(second=45))

    assert len(first_runs) == 1
    assert second_runs == []

    await controller_service.stop_feed()


@pytest.mark.asyncio
async def test_due_schedule_records_rejected_run_when_feeder_is_busy(controller_service):
    now = datetime(2026, 5, 22, 8, 30, 15)
    await controller_service.start_manual_feed(duration_seconds=1)
    schedule = await controller_service.create_schedule(
        name="Morning feed",
        enabled=True,
        weekdays=[WEEKDAYS[now.weekday()]],
        times=["08:30"],
        duration_seconds=1,
    )

    runs = await controller_service.run_due_schedules(now)

    assert len(runs) == 1
    assert runs[0].status == FeedRunStatus.REJECTED
    assert runs[0].schedule_id == schedule.id
    assert controller_service.store.has_feed_run_for_schedule_slot(
        schedule.id,
        datetime(2026, 5, 22, 8, 30),
    )

    await controller_service.stop_feed()
