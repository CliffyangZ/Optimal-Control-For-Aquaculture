"""Domain models for schedules, feed runs, and controller state."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class Weekday(str, Enum):
    """Supported weekdays."""

    MON = "mon"
    TUE = "tue"
    WED = "wed"
    THU = "thu"
    FRI = "fri"
    SAT = "sat"
    SUN = "sun"


class ActionType(str, Enum):
    """Supported feeding actions."""

    FEED = "feed"


class FeedRunStatus(str, Enum):
    """Feed run execution status."""

    SUCCESS = "success"
    FAILED = "failed"
    MISSED = "missed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class FeederState(str, Enum):
    """Current feeder state."""

    IDLE = "idle"
    ACTIVE = "active"


class ScheduleLoadState(str, Enum):
    """Schedule persistence state."""

    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class DriverMode(str, Enum):
    """Feeder driver mode."""

    SIMULATED = "simulated"
    HARDWARE = "hardware"


class FeedRunSource(str, Enum):
    """Feed run source."""

    MANUAL = "manual"
    SCHEDULE = "schedule"


@dataclass
class Schedule:
    """Feeding schedule entity."""

    id: str
    name: str
    enabled: bool
    action_type: ActionType
    weekdays: list[Weekday]
    times: list[str]  # HH:MM format
    duration_seconds: int
    note: Optional[str] = None
    next_run_at: Optional[datetime] = None
    version: int = 1
    deleted_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def new(
        cls,
        name: str,
        enabled: bool,
        action_type: ActionType,
        weekdays: list[Weekday],
        times: list[str],
        duration_seconds: int,
        note: Optional[str] = None,
    ) -> "Schedule":
        """Create a new schedule."""
        return cls(
            id=f"sch_{uuid4().hex[:12]}",
            name=name,
            enabled=enabled,
            action_type=action_type,
            weekdays=weekdays,
            times=times,
            duration_seconds=duration_seconds,
            note=note,
        )


@dataclass
class ScheduleSlot:
    """Normalized schedule slot for conflict detection."""

    schedule_id: str
    weekday: Weekday
    time: str  # HH:MM
    enabled: bool
    deleted: bool


@dataclass
class FeedRun:
    """Recorded feeding attempt or missed feeding."""

    id: str
    source: FeedRunSource
    status: FeedRunStatus
    duration_seconds: int
    driver_mode: DriverMode
    schedule_id: Optional[str] = None
    captured_schedule_name: Optional[str] = None
    planned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def new(
        cls,
        source: FeedRunSource,
        status: FeedRunStatus,
        duration_seconds: int,
        driver_mode: DriverMode,
        schedule_id: Optional[str] = None,
        captured_schedule_name: Optional[str] = None,
        planned_at: Optional[datetime] = None,
        message: Optional[str] = None,
    ) -> "FeedRun":
        """Create a new feed run record."""
        return cls(
            id=f"run_{uuid4().hex[:12]}",
            source=source,
            status=status,
            duration_seconds=duration_seconds,
            driver_mode=driver_mode,
            schedule_id=schedule_id,
            captured_schedule_name=captured_schedule_name,
            planned_at=planned_at,
            message=message,
        )


@dataclass
class ControllerActivity:
    """Controller lifecycle event for restart detection."""

    id: str
    event_type: str  # startup, shutdown, heartbeat
    recorded_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def startup(cls) -> "ControllerActivity":
        """Create a startup event."""
        return cls(id=f"act_{uuid4().hex[:12]}", event_type="startup")

    @classmethod
    def shutdown(cls) -> "ControllerActivity":
        """Create a shutdown event."""
        return cls(id=f"act_{uuid4().hex[:12]}", event_type="shutdown")

    @classmethod
    def heartbeat(cls) -> "ControllerActivity":
        """Create a heartbeat event."""
        return cls(id=f"act_{uuid4().hex[:12]}", event_type="heartbeat")


@dataclass
class ControllerStatus:
    """Current controller state snapshot."""

    controller_time: datetime
    timezone: str
    clock_warning: Optional[str]
    feeder_state: FeederState
    driver_mode: DriverMode
    schedule_load_state: ScheduleLoadState
    active_run: Optional[FeedRun]
    latest_warning: Optional[str]
    last_activity_at: Optional[datetime]
    state_version: int = 1
