"""Pydantic schemas for REST API and WebSocket messages."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from pi_controller.domain.models import (
    ActionType,
    DriverMode,
    FeederState,
    FeedRunSource,
    FeedRunStatus,
    ScheduleLoadState,
    Weekday,
)


# REST Request/Response Models


class ScheduleCreateRequest(BaseModel):
    """Request to create a schedule."""

    name: str
    enabled: bool = True
    action_type: ActionType = ActionType.FEED
    weekdays: list[Weekday]
    times: list[str]
    duration_seconds: int
    note: Optional[str] = None


class ScheduleUpdateRequest(BaseModel):
    """Request to update a schedule."""

    version: int
    name: str
    enabled: bool
    weekdays: list[Weekday]
    times: list[str]
    duration_seconds: int
    note: Optional[str] = None


class ScheduleEnableRequest(BaseModel):
    """Request to enable or disable a schedule."""

    version: int
    enabled: bool


class ScheduleDeleteRequest(BaseModel):
    """Request to delete a schedule."""

    version: int
    schedule_name_confirmation: str


class FeedRunResponse(BaseModel):
    """Feed run record response."""

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
    created_at: datetime


class ScheduleResponse(BaseModel):
    """Schedule response with derived fields."""

    id: str
    name: str
    enabled: bool
    action_type: ActionType
    weekdays: list[Weekday]
    times: list[str]
    duration_seconds: int
    note: Optional[str]
    next_run_at: Optional[datetime]
    latest_result: Optional[FeedRunResponse]
    version: int
    created_at: datetime
    updated_at: datetime


class ScheduleListResponse(BaseModel):
    """List of schedules."""

    schedules: list[ScheduleResponse]


class ControllerConfigResponse(BaseModel):
    """Controller configuration and defaults."""

    controller_id: str
    local_trusted_network_only: bool
    driver_mode: DriverMode
    default_scheduled_duration_seconds: int
    max_scheduled_duration_seconds: int
    default_manual_duration_seconds: int
    max_manual_duration_seconds: int
    supported_action_types: list[str]
    api_base_path: str
    ws_path: str


class ControllerStatusResponse(BaseModel):
    """Current controller status."""

    schema: str = "controller.status.v1"
    controller_time: datetime
    timezone: str
    clock_warning: Optional[str]
    feeder_state: FeederState
    driver_mode: DriverMode
    schedule_load_state: ScheduleLoadState
    active_run: Optional[FeedRunResponse]
    latest_warning: Optional[str]
    last_activity_at: Optional[datetime]
    state_version: int


class ControllerSnapshotResponse(BaseModel):
    """One-call laptop client snapshot data."""

    config: ControllerConfigResponse
    status: ControllerStatusResponse
    schedules: list[ScheduleResponse]
    recent_runs: list[FeedRunResponse]


class ManualFeedRequest(BaseModel):
    """Request to start a manual feed."""

    duration_seconds: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    message: str
    field: Optional[str] = None
    details: Optional[dict] = None


# WebSocket Message Models


class WebSocketMessage(BaseModel):
    """Base WebSocket message."""

    type: str
    client_id: str


class ControllerHello(WebSocketMessage):
    """Client subscription message."""

    type: str = "controller.hello"


class ControllerHeartbeat(WebSocketMessage):
    """Client keepalive message."""

    type: str = "controller.heartbeat"


class ControllerStop(WebSocketMessage):
    """Client unsubscription message."""

    type: str = "controller.stop"


class WebSocketEvent(BaseModel):
    """Base WebSocket event (server to client)."""

    schema: str
    type: str
    timestamp_ms: int
    state_version: int


class ControllerSnapshot(WebSocketEvent):
    """Full controller snapshot event."""

    schema: str = "controller.snapshot.v1"
    type: str = "controller.snapshot"
    snapshot: ControllerSnapshotResponse


class ControllerStatusEvent(WebSocketEvent):
    """Controller status changed event."""

    schema: str = "controller.status.v1"
    type: str = "controller.status"
    status: ControllerStatusResponse


class SchedulesChanged(WebSocketEvent):
    """Schedules changed event."""

    schema: str = "schedules.changed.v1"
    type: str = "schedules.changed"
    reason: str  # created, updated, enabled, disabled, deleted
    schedule_id: Optional[str] = None


class FeedRunStarted(WebSocketEvent):
    """Feed run started event."""

    schema: str = "feed_run.started.v1"
    type: str = "feed_run.started"
    feed_run: FeedRunResponse


class FeedRunFinished(WebSocketEvent):
    """Feed run finished event."""

    schema: str = "feed_run.finished.v1"
    type: str = "feed_run.finished"
    feed_run: FeedRunResponse


class HistoryChanged(WebSocketEvent):
    """History changed event."""

    schema: str = "history.changed.v1"
    type: str = "history.changed"
    feed_run: FeedRunResponse


class ControllerWarning(WebSocketEvent):
    """Controller warning event."""

    schema: str = "controller.warning.v1"
    type: str = "controller.warning"
    warning: str


class ControllerError(WebSocketEvent):
    """Controller error event."""

    schema: str = "controller.error.v1"
    type: str = "controller.error"
    error: str
    message: str
