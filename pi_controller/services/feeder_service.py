"""Single-active-feed state machine and cancellation behavior."""

import asyncio
from datetime import datetime
from typing import Optional

from pi_controller.domain.models import (
    DriverMode,
    FeedRun,
    FeedRunSource,
    FeedRunStatus,
)
from pi_controller.domain.validation import validate_manual_duration
from pi_controller.drivers.base import FeederDriver
from pi_controller.storage.sqlite_store import SQLiteStore
from pi_controller.config import config


class FeederService:
    """Manages feed execution state machine and timing."""

    def __init__(self, driver: FeederDriver, store: SQLiteStore) -> None:
        """Initialize the feeder service."""
        self.driver = driver
        self.store = store
        self.active_run: Optional[FeedRun] = None
        self.run_task: Optional[asyncio.Task] = None

    def get_active_run(self) -> Optional[FeedRun]:
        """Get the currently active feed run."""
        return self.active_run

    async def start_feed(
        self,
        source: FeedRunSource,
        duration_seconds: int,
        schedule_id: Optional[str] = None,
        captured_schedule_name: Optional[str] = None,
        planned_at: Optional[datetime] = None,
    ) -> FeedRun:
        """Start a feed run."""
        if self.active_run:
            raise ValueError("Feeder already running")

        if source == FeedRunSource.MANUAL:
            validate_manual_duration(duration_seconds)

        # Activate feeder
        activation_ok = await self.driver.activate()
        if not activation_ok:
            feed_run = FeedRun.new(
                source=source,
                status=FeedRunStatus.FAILED,
                duration_seconds=duration_seconds,
                driver_mode=DriverMode(config.DRIVER_MODE),
                schedule_id=schedule_id,
                captured_schedule_name=captured_schedule_name,
                planned_at=planned_at,
                message="Failed to activate feeder",
            )
            feed_run.started_at = datetime.utcnow()
            feed_run.finished_at = datetime.utcnow()
            self.store.create_feed_run(feed_run)
            return feed_run

        # Create run record
        feed_run = FeedRun.new(
            source=source,
            status=FeedRunStatus.SUCCESS,
            duration_seconds=duration_seconds,
            driver_mode=DriverMode(config.DRIVER_MODE),
            schedule_id=schedule_id,
            captured_schedule_name=captured_schedule_name,
            planned_at=planned_at,
        )
        feed_run.started_at = datetime.utcnow()
        self.active_run = feed_run

        # Schedule deactivation after duration
        self.run_task = asyncio.create_task(
            self._run_for_duration(duration_seconds)
        )

        return feed_run

    async def stop_feed(self) -> Optional[FeedRun]:
        """Stop the active feed."""
        if not self.active_run:
            return None

        feed_run = self.active_run
        feed_run.status = FeedRunStatus.CANCELLED
        feed_run.finished_at = datetime.utcnow()

        # Deactivate feeder
        await self.driver.deactivate()

        # Cancel timeout task
        if self.run_task:
            self.run_task.cancel()
            self.run_task = None

        # Save record
        self.store.create_feed_run(feed_run)
        self.store.prune_feed_runs(config.FEED_RUN_RETENTION_COUNT)

        self.active_run = None
        return feed_run

    async def _run_for_duration(self, duration_seconds: int) -> None:
        """Run the feeder for the specified duration."""
        try:
            await asyncio.sleep(duration_seconds)

            if self.active_run:
                # Normal completion
                await self.driver.deactivate()
                self.active_run.finished_at = datetime.utcnow()
                self.store.create_feed_run(self.active_run)
                self.store.prune_feed_runs(config.FEED_RUN_RETENTION_COUNT)
                self.active_run = None
        except asyncio.CancelledError:
            # Cancelled by stop_feed
            pass

    async def shutdown(self) -> None:
        """Gracefully shutdown the feeder service."""
        if self.active_run:
            await self.stop_feed()
