"""FastAPI controller app with REST endpoints and WebSocket events."""

import asyncio
import json
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from pi_controller.config import config
from pi_controller.domain.validation import ValidationError
from pi_controller.drivers.simulated import SimulatedFeederDriver
from pi_controller.drivers.gpio import GPIOFeederDriver
from pi_controller.schemas import (
    ControllerConfigResponse,
    ControllerSnapshotResponse,
    ControllerStatusResponse,
    FeedRunResponse,
    ManualFeedRequest,
    ScheduleCreateRequest,
    ScheduleDeleteRequest,
    ScheduleEnableRequest,
    ScheduleListResponse,
    ScheduleResponse,
    ScheduleUpdateRequest,
)
from pi_controller.services.controller_service import ControllerService
from pi_controller.services.realtime_hub import RealtimeHub
from pi_controller.storage.sqlite_store import SQLiteStore


# Initialize components
config.ensure_storage()
store = SQLiteStore(config.DATABASE_PATH)
realtime_hub = RealtimeHub()

# Initialize feeder driver
if config.DRIVER_MODE == "hardware":
    feeder_driver = GPIOFeederDriver()
else:
    feeder_driver = SimulatedFeederDriver()

controller_service = ControllerService(
    store=store,
    feeder_driver=feeder_driver,
    realtime_hub=realtime_hub,
)

# Create FastAPI app
app = FastAPI(title="Pi Scheduled Controller")

# CORS configuration (local trusted network only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup and shutdown
@app.on_event("startup")
async def startup() -> None:
    """Initialize controller on startup."""
    await controller_service.record_activity("startup")
    controller_service.start_scheduler()


@app.on_event("shutdown")
async def shutdown() -> None:
    """Cleanup on shutdown."""
    await controller_service.stop_scheduler()
    await controller_service.feeder_service.shutdown()
    await controller_service.record_activity("shutdown")


# REST Endpoints

@app.get("/controller/config", response_model=ControllerConfigResponse)
async def get_config() -> dict:
    """Get controller configuration."""
    return controller_service.get_config_dict()


@app.get("/controller/status", response_model=ControllerStatusResponse)
async def get_status() -> dict:
    """Get current controller status."""
    status = await controller_service.get_status()
    return {
        "schema": "controller.status.v1",
        "controller_time": status.controller_time,
        "timezone": status.timezone,
        "clock_warning": status.clock_warning,
        "feeder_state": status.feeder_state.value,
        "driver_mode": status.driver_mode.value,
        "schedule_load_state": status.schedule_load_state.value,
        "active_run": _feed_run_to_response(status.active_run) if status.active_run else None,
        "latest_warning": status.latest_warning,
        "last_activity_at": status.last_activity_at,
        "state_version": status.state_version,
    }


@app.get("/controller/snapshot", response_model=ControllerSnapshotResponse)
async def get_snapshot() -> dict:
    """Get laptop client bootstrap snapshot."""
    snapshot = await controller_service.get_snapshot()
    return {
        "config": snapshot["config"],
        "status": await _status_to_response(snapshot["status"]),
        "schedules": [_schedule_to_response(s, store) for s in snapshot["schedules"]],
        "recent_runs": [_feed_run_to_response(r) for r in snapshot["recent_runs"]],
    }


@app.get("/controller/schedules", response_model=ScheduleListResponse)
async def get_schedules() -> dict:
    """Get all visible schedules."""
    schedules = await controller_service.get_schedules()
    return {
        "schedules": [_schedule_to_response(s, store) for s in schedules],
    }


@app.post("/controller/schedules", response_model=ScheduleResponse, status_code=201)
async def create_schedule(req: ScheduleCreateRequest) -> dict:
    """Create a new schedule."""
    try:
        schedule = await controller_service.create_schedule(
            name=req.name,
            enabled=req.enabled,
            weekdays=req.weekdays,
            times=req.times,
            duration_seconds=req.duration_seconds,
            note=req.note,
        )
        return _schedule_to_response(schedule, store)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.error_code,
                "message": e.message,
                "field": e.field,
                "details": e.details,
            },
        )


@app.put("/controller/schedules/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    schedule_id: str,
    req: ScheduleUpdateRequest,
) -> dict:
    """Update a schedule."""
    try:
        schedule = await controller_service.update_schedule(
            schedule_id=schedule_id,
            version=req.version,
            name=req.name,
            enabled=req.enabled,
            weekdays=req.weekdays,
            times=req.times,
            duration_seconds=req.duration_seconds,
            note=req.note,
        )
        return _schedule_to_response(schedule, store)
    except ValidationError as e:
        status_code = 409 if "stale" in e.error_code else 400
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "field": e.field,
                "details": e.details,
            },
        )


@app.patch("/controller/schedules/{schedule_id}/enabled")
async def set_schedule_enabled(
    schedule_id: str,
    req: ScheduleEnableRequest,
) -> dict:
    """Enable or disable a schedule."""
    try:
        schedule = await controller_service.set_schedule_enabled(
            schedule_id=schedule_id,
            version=req.version,
            enabled=req.enabled,
        )
        return _schedule_to_response(schedule, store)
    except ValidationError as e:
        status_code = 409 if "stale" in e.error_code else 400
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "field": e.field,
                "details": e.details,
            },
        )


@app.delete("/controller/schedules/{schedule_id}", status_code=204)
async def delete_schedule(
    schedule_id: str,
    req: ScheduleDeleteRequest,
) -> None:
    """Delete a schedule."""
    try:
        await controller_service.delete_schedule(
            schedule_id=schedule_id,
            version=req.version,
            schedule_name_confirmation=req.schedule_name_confirmation,
        )
    except ValidationError as e:
        status_code = 409 if "stale" in e.error_code else 400
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "field": e.field,
                "details": e.details,
            },
        )


@app.post("/controller/feed-runs/manual", response_model=FeedRunResponse, status_code=201)
async def start_manual_feed(req: ManualFeedRequest) -> dict:
    """Start a manual feed."""
    try:
        feed_run = await controller_service.start_manual_feed(
            duration_seconds=req.duration_seconds,
        )
        return _feed_run_to_response(feed_run)
    except (ValueError, ValidationError) as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "feeder_already_running" if isinstance(e, ValueError) else "invalid_duration",
                "message": str(e),
            },
        )


@app.post("/controller/feed-runs/stop", status_code=204)
async def stop_feed() -> None:
    """Stop the active feed."""
    await controller_service.stop_feed()


@app.get("/controller/feed-runs", response_model=list[FeedRunResponse])
async def get_feed_runs(limit: int = Query(20, ge=1, le=1000)) -> list[dict]:
    """Get recent feed run history."""
    runs = await controller_service.get_recent_runs(limit=limit)
    return [_feed_run_to_response(r) for r in runs]


# WebSocket

@app.websocket("/ws/controller")
async def websocket_controller(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    client_id = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "controller.hello":
                client_id = message.get("client_id")
                queue = await realtime_hub.subscribe(client_id)

                # Send snapshot
                snapshot = await controller_service.get_snapshot()
                from pi_controller.schemas import ControllerSnapshot

                event = ControllerSnapshot(
                    state_version=controller_service.state_version,
                    timestamp_ms=int(datetime.utcnow().timestamp() * 1000),
                    snapshot={
                        "config": snapshot["config"],
                        "status": await _status_to_response(snapshot["status"]),
                        "schedules": [
                            _schedule_to_response(s, store)
                            for s in snapshot["schedules"]
                        ],
                        "recent_runs": [
                            _feed_run_to_response(r)
                            for r in snapshot["recent_runs"]
                        ],
                    },
                )
                await websocket.send_json(event.model_dump())

                # Send queued events
                async def send_queue_messages():
                    while True:
                        try:
                            msg = queue.get_nowait()
                            await websocket.send_text(msg)
                        except asyncio.QueueEmpty:
                            await asyncio.sleep(0.1)

                await send_queue_messages()

            elif msg_type == "controller.heartbeat":
                pass

            elif msg_type == "controller.stop":
                break

    except WebSocketDisconnect:
        pass
    finally:
        if client_id:
            await realtime_hub.unsubscribe(client_id)
        await websocket.close()


# Service descriptor

@app.get("/")
async def service_descriptor() -> dict:
    """Describe the Raspberry Pi controller service for laptop clients."""
    return {
        "service": "pi-scheduled-controller",
        "api_base_path": "/controller",
        "ws_path": "/ws/controller",
    }


# Helpers

def _schedule_to_response(schedule, store: SQLiteStore) -> dict:
    """Convert a Schedule model to API response."""
    latest_run = store.get_latest_run_for_schedule(schedule.id)
    return {
        "id": schedule.id,
        "name": schedule.name,
        "enabled": schedule.enabled,
        "action_type": schedule.action_type.value,
        "weekdays": [w.value for w in schedule.weekdays],
        "times": schedule.times,
        "duration_seconds": schedule.duration_seconds,
        "note": schedule.note,
        "next_run_at": schedule.next_run_at,
        "latest_result": _feed_run_to_response(latest_run) if latest_run else None,
        "version": schedule.version,
        "created_at": schedule.created_at,
        "updated_at": schedule.updated_at,
    }


def _feed_run_to_response(feed_run) -> dict:
    """Convert a FeedRun model to API response."""
    return {
        "id": feed_run.id,
        "source": feed_run.source.value,
        "status": feed_run.status.value,
        "duration_seconds": feed_run.duration_seconds,
        "driver_mode": feed_run.driver_mode.value,
        "schedule_id": feed_run.schedule_id,
        "captured_schedule_name": feed_run.captured_schedule_name,
        "planned_at": feed_run.planned_at,
        "started_at": feed_run.started_at,
        "finished_at": feed_run.finished_at,
        "message": feed_run.message,
        "created_at": feed_run.created_at,
    }


async def _status_to_response(status):
    """Convert a ControllerStatus model to API response."""
    return {
        "schema": "controller.status.v1",
        "controller_time": status.controller_time,
        "timezone": status.timezone,
        "clock_warning": status.clock_warning,
        "feeder_state": status.feeder_state.value,
        "driver_mode": status.driver_mode.value,
        "schedule_load_state": status.schedule_load_state.value,
        "active_run": _feed_run_to_response(status.active_run) if status.active_run else None,
        "latest_warning": status.latest_warning,
        "last_activity_at": status.last_activity_at,
        "state_version": status.state_version,
    }
