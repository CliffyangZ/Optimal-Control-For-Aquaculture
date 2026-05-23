"""SQLite persistence layer for controller state."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from pi_controller.domain.models import (
    ActionType,
    ControllerActivity,
    DriverMode,
    FeedRun,
    FeedRunSource,
    FeedRunStatus,
    Schedule,
    ScheduleSlot,
    Weekday,
)
from pi_controller.storage.migrations import create_schema


class SQLiteStore:
    """SQLite-backed persistence store."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the store and create schema if needed."""
        self.db_path = db_path
        create_schema(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # Schedules

    def create_schedule(self, schedule: Schedule) -> Schedule:
        """Create a new schedule."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO schedules
            (id, name, enabled, action_type, weekdays, times, duration_seconds, note,
             version, deleted_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schedule.id,
                schedule.name,
                schedule.enabled,
                schedule.action_type.value,
                json.dumps([w.value for w in schedule.weekdays]),
                json.dumps(schedule.times),
                schedule.duration_seconds,
                schedule.note,
                schedule.version,
                schedule.deleted_at.isoformat() if schedule.deleted_at else None,
                schedule.created_at.isoformat(),
                schedule.updated_at.isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        return schedule

    def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Get a schedule by ID (including soft-deleted)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,))
        row = cursor.fetchone()
        conn.close()
        return self._row_to_schedule(row) if row else None

    def get_visible_schedules(self) -> list[Schedule]:
        """Get non-deleted schedules."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM schedules WHERE deleted_at IS NULL ORDER BY created_at")
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_schedule(row) for row in rows]

    def update_schedule(self, schedule: Schedule) -> Schedule:
        """Update a schedule."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE schedules
            SET name = ?, enabled = ?, weekdays = ?, times = ?, duration_seconds = ?,
                note = ?, version = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                schedule.name,
                schedule.enabled,
                json.dumps([w.value for w in schedule.weekdays]),
                json.dumps(schedule.times),
                schedule.duration_seconds,
                schedule.note,
                schedule.version,
                schedule.updated_at.isoformat(),
                schedule.id,
            ),
        )
        conn.commit()
        conn.close()
        return schedule

    def soft_delete_schedule(self, schedule_id: str, deleted_at: datetime) -> None:
        """Soft delete a schedule."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE schedules SET deleted_at = ? WHERE id = ?",
            (deleted_at.isoformat(), schedule_id),
        )
        conn.commit()
        conn.close()

    def get_enabled_slots(self) -> list[tuple[str, Weekday, str]]:
        """Get all enabled non-deleted schedule slots as (schedule_id, weekday, time)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT schedule_id, weekday, time FROM schedule_slots
            WHERE enabled = 1 AND deleted = 0
            """,
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            (row[0], Weekday(row[1]), row[2])
            for row in rows
        ]

    def update_schedule_slots(self, schedule_id: str, weekdays: list[Weekday], times: list[str], enabled: bool, deleted: bool) -> None:
        """Update normalized schedule slots for conflict detection."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Delete old slots for this schedule
        cursor.execute("DELETE FROM schedule_slots WHERE schedule_id = ?", (schedule_id,))

        # Insert new slots
        for weekday in weekdays:
            for time in times:
                cursor.execute(
                    """
                    INSERT INTO schedule_slots
                    (schedule_id, weekday, time, enabled, deleted)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (schedule_id, weekday.value, time, enabled, deleted),
                )

        conn.commit()
        conn.close()

    # Feed runs

    def create_feed_run(self, feed_run: FeedRun) -> FeedRun:
        """Create a feed run record."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO feed_runs
            (id, source, status, duration_seconds, driver_mode, schedule_id,
             captured_schedule_name, planned_at, started_at, finished_at, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feed_run.id,
                feed_run.source.value,
                feed_run.status.value,
                feed_run.duration_seconds,
                feed_run.driver_mode.value,
                feed_run.schedule_id,
                feed_run.captured_schedule_name,
                feed_run.planned_at.isoformat() if feed_run.planned_at else None,
                feed_run.started_at.isoformat() if feed_run.started_at else None,
                feed_run.finished_at.isoformat() if feed_run.finished_at else None,
                feed_run.message,
                feed_run.created_at.isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        return feed_run

    def get_recent_runs(self, limit: int = 20) -> list[FeedRun]:
        """Get recent feed runs (newest first)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM feed_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_feed_run(row) for row in rows]

    def get_latest_run_for_schedule(self, schedule_id: str) -> Optional[FeedRun]:
        """Get the latest run for a schedule."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM feed_runs
            WHERE schedule_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (schedule_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return self._row_to_feed_run(row) if row else None

    def has_feed_run_for_schedule_slot(self, schedule_id: str, planned_at: datetime) -> bool:
        """Return whether a schedule slot has already been attempted."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM feed_runs
            WHERE schedule_id = ? AND planned_at = ?
            LIMIT 1
            """,
            (schedule_id, planned_at.isoformat()),
        )
        row = cursor.fetchone()
        conn.close()
        return row is not None

    def prune_feed_runs(self, retention_count: int) -> None:
        """Keep only the latest N feed runs."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM feed_runs
            WHERE id NOT IN (
                SELECT id FROM feed_runs
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            (retention_count,),
        )
        conn.commit()
        conn.close()

    # Controller activity

    def record_activity(self, activity: ControllerActivity) -> None:
        """Record a controller lifecycle event."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO controller_activity (id, event_type, recorded_at)
            VALUES (?, ?, ?)
            """,
            (activity.id, activity.event_type, activity.recorded_at.isoformat()),
        )
        conn.commit()
        conn.close()

    def get_last_activity(self) -> Optional[ControllerActivity]:
        """Get the most recent activity record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM controller_activity
            ORDER BY recorded_at DESC
            LIMIT 1
            """,
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return ControllerActivity(
            id=row["id"],
            event_type=row["event_type"],
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
        )

    # Helpers

    def _row_to_schedule(self, row: sqlite3.Row) -> Schedule:
        """Convert a database row to a Schedule."""
        return Schedule(
            id=row["id"],
            name=row["name"],
            enabled=bool(row["enabled"]),
            action_type=ActionType(row["action_type"]),
            weekdays=[Weekday(w) for w in json.loads(row["weekdays"])],
            times=json.loads(row["times"]),
            duration_seconds=row["duration_seconds"],
            note=row["note"],
            next_run_at=datetime.fromisoformat(row["next_run_at"]) if row["next_run_at"] else None,
            version=row["version"],
            deleted_at=datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_feed_run(self, row: sqlite3.Row) -> FeedRun:
        """Convert a database row to a FeedRun."""
        return FeedRun(
            id=row["id"],
            source=FeedRunSource(row["source"]),
            status=FeedRunStatus(row["status"]),
            duration_seconds=row["duration_seconds"],
            driver_mode=DriverMode(row["driver_mode"]),
            schedule_id=row["schedule_id"],
            captured_schedule_name=row["captured_schedule_name"],
            planned_at=datetime.fromisoformat(row["planned_at"]) if row["planned_at"] else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            message=row["message"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
