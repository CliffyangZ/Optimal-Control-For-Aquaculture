"""SQLite schema creation and lightweight migration helpers."""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1


def create_schema(db_path: Path) -> None:
    """Create the controller SQLite schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Metadata table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS _schema (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    # Check if already initialized
    cursor.execute("SELECT value FROM _schema WHERE key = 'version'")
    result = cursor.fetchone()
    if result:
        conn.close()
        return

    # Schedules
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS schedules (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            enabled BOOLEAN NOT NULL,
            action_type TEXT NOT NULL,
            weekdays TEXT NOT NULL,
            times TEXT NOT NULL,
            duration_seconds INTEGER NOT NULL,
            note TEXT,
            next_run_at TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            deleted_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # Schedule slots (normalized for conflict detection)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS schedule_slots (
            schedule_id TEXT NOT NULL,
            weekday TEXT NOT NULL,
            time TEXT NOT NULL,
            enabled BOOLEAN NOT NULL,
            deleted BOOLEAN NOT NULL,
            PRIMARY KEY (schedule_id, weekday, time),
            FOREIGN KEY (schedule_id) REFERENCES schedules(id)
        )
        """
    )

    # Feed runs
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feed_runs (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            status TEXT NOT NULL,
            duration_seconds INTEGER NOT NULL,
            driver_mode TEXT NOT NULL,
            schedule_id TEXT,
            captured_schedule_name TEXT,
            planned_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            message TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (schedule_id) REFERENCES schedules(id)
        )
        """
    )

    # Controller activity (for restart detection)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS controller_activity (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        )
        """
    )

    # Indexes
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_schedules_enabled_deleted ON schedules(enabled, deleted_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_schedule_slots_enabled ON schedule_slots(enabled, deleted)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feed_runs_created ON feed_runs(created_at DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feed_runs_schedule ON feed_runs(schedule_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_activity_type ON controller_activity(event_type)"
    )

    # Mark schema as initialized
    cursor.execute(
        "INSERT INTO _schema (key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION)),
    )

    conn.commit()
    conn.close()


def get_schema_version(db_path: Path) -> int:
    """Get the current schema version."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM _schema WHERE key = 'version'")
    result = cursor.fetchone()
    conn.close()
    return int(result[0]) if result else 0
