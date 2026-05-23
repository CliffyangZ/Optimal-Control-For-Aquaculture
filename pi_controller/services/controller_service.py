"""Service layer with use cases for schedules, feed runs, activity, and snapshots."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from pi_controller.config import config
from pi_controller.domain.models import (
    ActionType,
    ControllerActivity,
    ControllerStatus,
    DriverMode,
    FeederState,
    FeedRun,
    FeedRunSource,
    FeedRunStatus,
    Schedule,
    ScheduleLoadState,
)
from pi_controller.domain.validation import (
    ValidationError,
    validate_create_request,
    validate_delete_confirmation,
    validate_slot_conflicts,
    validate_update_request,
)
from pi_controller.drivers.base import FeederDriver
from pi_controller.services.feeder_service import FeederService
from pi_controller.services.realtime_hub import RealtimeHub
from pi_controller.storage.sqlite_store import SQLiteStore


class ControllerService:
    """Main controller service coordinating all use cases."""

    def __init__(
        self,
        store: SQLiteStore,
        feeder_driver: FeederDriver,
        realtime_hub: RealtimeHub,
    ) -> None:
        """Initialize the controller service."""
        self.store = store
        self.feeder_driver = feeder_driver
        self.realtime_hub = realtime_hub
        self.feeder_service = FeederService(feeder_driver, store)
        self.state_version = 1
        self.last_activity_at: Optional[datetime] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._attempted_schedule_slots: set[tuple[str, datetime]] = set()

    # Configuration

    def get_config_dict(self) -> dict:
        """Get controller configuration for REST response."""
        return {
            "controller_id": config.CONTROLLER_ID,
            "local_trusted_network_only": config.LOCAL_TRUSTED_NETWORK_ONLY,
            "driver_mode": config.DRIVER_MODE,
            "default_scheduled_duration_seconds": config.DEFAULT_SCHEDULED_DURATION_SECONDS,
            "max_scheduled_duration_seconds": config.MAX_SCHEDULED_DURATION_SECONDS,
            "default_manual_duration_seconds": config.DEFAULT_MANUAL_DURATION_SECONDS,
            "max_manual_duration_seconds": config.MAX_MANUAL_DURATION_SECONDS,
            "supported_action_types": ["feed"],
            "api_base_path": "/controller",
            "ws_path": "/ws/controller",
        }

    # Scheduler

    def start_scheduler(self) -> None:
        """Start the background schedule runner."""
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop_scheduler(self) -> None:
        """Stop the background schedule runner."""
        if not self._scheduler_task:
            return
        self._scheduler_task.cancel()
        try:
            await self._scheduler_task
        except asyncio.CancelledError:
            pass
        self._scheduler_task = None

    async def _scheduler_loop(self) -> None:
        """Poll schedules and trigger due feeding runs."""
        while True:
            await self.run_due_schedules(datetime.utcnow())
            await asyncio.sleep(config.SCHEDULE_TICK_SECONDS)

    async def run_due_schedules(self, now: datetime) -> list[FeedRun]:
        """Run enabled schedules whose weekday/time match the current minute."""
        planned_at = now.replace(second=0, microsecond=0)
        oldest_kept_slot = planned_at - timedelta(hours=config.MISSED_RUN_LOOKBACK_HOURS)
        self._attempted_schedule_slots = {
            slot_key
            for slot_key in self._attempted_schedule_slots
            if slot_key[1] >= oldest_kept_slot
        }
        current_weekday = _weekday_from_datetime(planned_at)
        current_time = planned_at.strftime("%H:%M")
        started_runs: list[FeedRun] = []

        for schedule in self.store.get_visible_schedules():
            if not schedule.enabled:
                continue
            if current_weekday not in schedule.weekdays or current_time not in schedule.times:
                continue

            slot_key = (schedule.id, planned_at)
            if slot_key in self._attempted_schedule_slots:
                continue
            if self.store.has_feed_run_for_schedule_slot(schedule.id, planned_at):
                self._attempted_schedule_slots.add(slot_key)
                continue

            self._attempted_schedule_slots.add(slot_key)
            try:
                feed_run = await self.feeder_service.start_feed(
                    source=FeedRunSource.SCHEDULE,
                    duration_seconds=schedule.duration_seconds,
                    schedule_id=schedule.id,
                    captured_schedule_name=schedule.name,
                    planned_at=planned_at,
                )
            except ValueError:
                feed_run = FeedRun.new(
                    source=FeedRunSource.SCHEDULE,
                    status=FeedRunStatus.REJECTED,
                    duration_seconds=schedule.duration_seconds,
                    driver_mode=DriverMode(config.DRIVER_MODE),
                    schedule_id=schedule.id,
                    captured_schedule_name=schedule.name,
                    planned_at=planned_at,
                    message="Feeder already running",
                )
                self.store.create_feed_run(feed_run)
                self.store.prune_feed_runs(config.FEED_RUN_RETENTION_COUNT)

            self.state_version += 1
            started_runs.append(feed_run)

        return started_runs

    # Status

    async def get_status(self) -> ControllerStatus:
        """Get current controller status."""
        now = datetime.now(timezone.utc)
        active_run = self.feeder_service.get_active_run()
        feeder_state = FeederState.ACTIVE if active_run else FeederState.IDLE

        return ControllerStatus(
            controller_time=now,
            timezone="UTC",
            clock_warning=None,
            feeder_state=feeder_state,
            driver_mode=DriverMode(config.DRIVER_MODE),
            schedule_load_state=ScheduleLoadState.LOADED,
            active_run=active_run,
            latest_warning=None,
            last_activity_at=self.last_activity_at,
            state_version=self.state_version,
        )

    async def record_activity(self, event_type: str) -> None:
        """Record a controller activity event."""
        activity = ControllerActivity(
            id=f"act_{self.state_version}",
            event_type=event_type,
            recorded_at=datetime.utcnow(),
        )
        self.store.record_activity(activity)
        self.last_activity_at = activity.recorded_at
        self.state_version += 1

    # Schedules

    async def create_schedule(
        self,
        name: str,
        enabled: bool,
        weekdays: list,
        times: list[str],
        duration_seconds: int,
        note: Optional[str] = None,
    ) -> Schedule:
        """Create a new schedule with validation."""
        validate_create_request(name, weekdays, times, duration_seconds)

        # Check for slot conflicts
        existing_slots = self.store.get_enabled_slots()
        new_schedule = Schedule.new(
            name=name,
            enabled=enabled,
            action_type=ActionType.FEED,
            weekdays=weekdays,
            times=times,
            duration_seconds=duration_seconds,
            note=note,
        )
        validate_slot_conflicts(new_schedule, [], existing_slots)

        # Save schedule
        schedule = self.store.create_schedule(new_schedule)
        self.store.update_schedule_slots(
            schedule.id, schedule.weekdays, schedule.times, schedule.enabled, False
        )

        self.state_version += 1
        await self.realtime_hub.broadcast(
            self._create_schedules_changed_event("created", schedule.id)
        )

        return schedule

    async def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Get a schedule by ID."""
        return self.store.get_schedule(schedule_id)

    async def get_schedules(self) -> list[Schedule]:
        """Get all visible schedules."""
        return self.store.get_visible_schedules()

    async def update_schedule(
        self,
        schedule_id: str,
        version: int,
        name: str,
        enabled: bool,
        weekdays: list,
        times: list[str],
        duration_seconds: int,
        note: Optional[str] = None,
    ) -> Schedule:
        """Update a schedule with stale version check."""
        schedule = self.store.get_schedule(schedule_id)
        if not schedule:
            raise ValidationError("schedule_not_found", "Schedule not found.")
        if schedule.version != version:
            raise ValidationError("stale_schedule_version", "Schedule was modified. Please refresh.")

        validate_update_request(name, weekdays, times, duration_seconds)

        # Check slot conflicts (excluding self)
        existing_slots = [
            (sid, wd, t)
            for sid, wd, t in self.store.get_enabled_slots()
            if sid != schedule_id
        ]
        new_schedule = Schedule(
            id=schedule.id,
            name=name,
            enabled=enabled,
            action_type=ActionType.FEED,
            weekdays=weekdays,
            times=times,
            duration_seconds=duration_seconds,
            note=note,
            version=schedule.version + 1,
            created_at=schedule.created_at,
            updated_at=datetime.utcnow(),
        )
        validate_slot_conflicts(new_schedule, [], existing_slots)

        # Save update
        schedule = self.store.update_schedule(new_schedule)
        self.store.update_schedule_slots(
            schedule.id, schedule.weekdays, schedule.times, schedule.enabled, False
        )

        self.state_version += 1
        await self.realtime_hub.broadcast(
            self._create_schedules_changed_event("updated", schedule.id)
        )

        return schedule

    async def set_schedule_enabled(
        self,
        schedule_id: str,
        version: int,
        enabled: bool,
    ) -> Schedule:
        """Enable or disable a schedule."""
        schedule = self.store.get_schedule(schedule_id)
        if not schedule:
            raise ValidationError("schedule_not_found", "Schedule not found.")
        if schedule.version != version:
            raise ValidationError("stale_schedule_version", "Schedule was modified. Please refresh.")

        if enabled:
            # Re-validate for conflicts if enabling
            existing_slots = [
                (sid, wd, t)
                for sid, wd, t in self.store.get_enabled_slots()
                if sid != schedule_id
            ]
            new_schedule = Schedule(
                id=schedule.id,
                name=schedule.name,
                enabled=True,
                action_type=schedule.action_type,
                weekdays=schedule.weekdays,
                times=schedule.times,
                duration_seconds=schedule.duration_seconds,
                note=schedule.note,
                version=schedule.version + 1,
                created_at=schedule.created_at,
                updated_at=datetime.utcnow(),
            )
            validate_slot_conflicts(new_schedule, [], existing_slots)

        schedule.enabled = enabled
        schedule.version += 1
        schedule.updated_at = datetime.utcnow()
        schedule = self.store.update_schedule(schedule)
        self.store.update_schedule_slots(
            schedule.id, schedule.weekdays, schedule.times, schedule.enabled, False
        )

        self.state_version += 1
        reason = "enabled" if enabled else "disabled"
        await self.realtime_hub.broadcast(
            self._create_schedules_changed_event(reason, schedule.id)
        )

        return schedule

    async def delete_schedule(
        self,
        schedule_id: str,
        version: int,
        schedule_name_confirmation: str,
    ) -> None:
        """Soft delete a schedule."""
        schedule = self.store.get_schedule(schedule_id)
        if not schedule:
            raise ValidationError("schedule_not_found", "Schedule not found.")
        if schedule.version != version:
            raise ValidationError("stale_schedule_version", "Schedule was modified. Please refresh.")

        validate_delete_confirmation(schedule.name, schedule_name_confirmation)

        self.store.soft_delete_schedule(schedule_id, datetime.utcnow())
        self.store.update_schedule_slots(
            schedule_id, schedule.weekdays, schedule.times, False, True
        )

        self.state_version += 1
        await self.realtime_hub.broadcast(
            self._create_schedules_changed_event("deleted", schedule.id)
        )

    # Feed runs

    async def start_manual_feed(self, duration_seconds: int) -> FeedRun:
        """Start a manual feed."""
        feed_run = await self.feeder_service.start_feed(
            source=FeedRunSource.MANUAL,
            duration_seconds=duration_seconds,
            schedule_id=None,
        )
        self.state_version += 1
        return feed_run

    async def stop_feed(self) -> Optional[FeedRun]:
        """Stop the active feed."""
        feed_run = await self.feeder_service.stop_feed()
        if feed_run:
            self.state_version += 1
        return feed_run

    async def get_recent_runs(self, limit: int = 20) -> list[FeedRun]:
        """Get recent feed run history."""
        return self.store.get_recent_runs(limit)

    # Snapshot

    async def get_snapshot(self) -> dict:
        """Get one-call laptop client snapshot data."""
        config = self.get_config_dict()
        status = await self.get_status()
        schedules = await self.get_schedules()
        recent_runs = await self.get_recent_runs(limit=20)

        return {
            "config": config,
            "status": status,
            "schedules": schedules,
            "recent_runs": recent_runs,
        }

    # Helpers

    def _create_schedules_changed_event(self, reason: str, schedule_id: Optional[str] = None):
        """Create a schedules.changed event."""
        from pi_controller.schemas import SchedulesChanged

        return SchedulesChanged(
            state_version=self.state_version,
            timestamp_ms=int(datetime.utcnow().timestamp() * 1000),
            reason=reason,
            schedule_id=schedule_id,
        )


def _weekday_from_datetime(value: datetime):
    """Map Python weekday numbers to controller weekday values."""
    from pi_controller.domain.models import Weekday

    return [
        Weekday.MON,
        Weekday.TUE,
        Weekday.WED,
        Weekday.THU,
        Weekday.FRI,
        Weekday.SAT,
        Weekday.SUN,
    ][value.weekday()]
