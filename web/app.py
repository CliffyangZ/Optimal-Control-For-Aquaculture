"""FastAPI backend for the low-latency aquaculture monitoring UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.setting import (
    BACKEND_PORT,
    FRONTEND_DIST_DIR,
    LOCAL_TRUSTED_NETWORK_ONLY,
    MEDIAMTX_HOST,
    MEDIAMTX_WEBRTC_PORT,
    SOURCE_ID,
    STALE_OVERLAY_MS,
    WEBRTC_PATH_TEMPLATE,
)
from core.models import StreamStatus, DetectionHealthStatus
from web.detection_worker import detection_worker
from web.overlay_hub import overlay_hub
from web.schemas import (
    RuntimeConfigResponse,
    SessionStatusResponse,
    StopSessionRequest,
)

logger = logging.getLogger(__name__)

LOOPBACK_MEDIAMTX_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}

app = FastAPI(title="Aquaculture Monitoring API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _browser_host(request: Request) -> str:
    return request.url.hostname or "localhost"


def _browser_visible_mediamtx_host(request: Request) -> str:
    if MEDIAMTX_HOST in LOOPBACK_MEDIAMTX_HOSTS:
        return _browser_host(request)
    return MEDIAMTX_HOST


def _build_config(request: Request) -> RuntimeConfigResponse:
    host = _browser_host(request)
    mediamtx_host = _browser_visible_mediamtx_host(request)
    webrtc_url = (
        f"http://{mediamtx_host}:{MEDIAMTX_WEBRTC_PORT}"
        f"{WEBRTC_PATH_TEMPLATE.format(source_id=SOURCE_ID)}"
    )
    ws_scheme = "wss" if request.url.scheme == "https" else "ws"
    overlay_ws_url = f"{ws_scheme}://{host}:{BACKEND_PORT}/ws/overlays"
    return RuntimeConfigResponse(
        source_id=SOURCE_ID,
        webrtc_url=webrtc_url,
        overlay_ws_url=overlay_ws_url,
        stale_overlay_ms=STALE_OVERLAY_MS,
        local_trusted_network_only=LOCAL_TRUSTED_NETWORK_ONLY,
    )


def _session_to_response(
    viewer_id: Optional[str] = None,
    message: Optional[str] = None,
) -> SessionStatusResponse:
    session = detection_worker.get_session_snapshot()
    if session is None:
        return SessionStatusResponse(
            source_id=SOURCE_ID,
            stream_status=StreamStatus.WAITING,
            detection_status=DetectionHealthStatus.UNAVAILABLE,
            active_viewer_count=0,
            viewer_id=viewer_id,
            message=message,
        )
    return SessionStatusResponse(
        source_id=session.source_id,
        stream_status=session.stream_status,
        detection_status=session.detection_status,
        active_viewer_count=session.active_viewer_count,
        started_at=session.started_at.isoformat() if session.started_at else None,
        last_frame_at=session.last_frame_at.isoformat() if session.last_frame_at else None,
        last_overlay_at=session.last_overlay_at.isoformat() if session.last_overlay_at else None,
        detection_fps=session.detection_fps,
        viewer_id=viewer_id,
        message=message,
    )


# --- REST endpoints ---

@app.get("/config", response_model=RuntimeConfigResponse)
async def get_config(request: Request) -> RuntimeConfigResponse:
    return _build_config(request)


@app.post("/session/start", response_model=SessionStatusResponse)
async def start_session() -> SessionStatusResponse:
    from uuid import uuid4
    viewer_id = str(uuid4())
    await detection_worker.register_viewer(viewer_id)
    return _session_to_response(viewer_id=viewer_id)


@app.post("/session/stop", response_model=SessionStatusResponse)
async def stop_session(body: Optional[StopSessionRequest] = None) -> SessionStatusResponse:
    viewer_id = body.viewer_id if body else None
    if viewer_id:
        await detection_worker.unregister_viewer(viewer_id)
    return _session_to_response()


@app.get("/status", response_model=SessionStatusResponse)
async def get_status() -> SessionStatusResponse:
    return _session_to_response()


# --- WebSocket endpoint ---

@app.websocket("/ws/overlays")
async def ws_overlays(websocket: WebSocket) -> None:
    await websocket.accept()
    viewer_id: Optional[str] = None
    push_task: Optional[asyncio.Task] = None

    try:
        while True:
            try:
                raw = await websocket.receive_text()
                data = json.loads(raw)
            except WebSocketDisconnect:
                break
            except Exception:
                break

            msg_type = data.get("type", "")

            if msg_type == "viewer.hello":
                viewer_id = data.get("viewer_id", "")
                # The viewer was already registered via POST /session/start.
                # Just subscribe to the overlay hub for this WS connection;
                # calling register_viewer again would reset the heartbeat timer
                # and re-run session-start logic unnecessarily.
                queue = overlay_hub.subscribe(viewer_id)
                push_task = asyncio.create_task(
                    _push_overlays(websocket, queue, viewer_id)
                )

            elif msg_type == "viewer.heartbeat":
                vid = data.get("viewer_id", viewer_id)
                if vid:
                    await detection_worker.update_heartbeat(vid)

            elif msg_type == "viewer.stop":
                vid = data.get("viewer_id", viewer_id)
                if vid:
                    await detection_worker.unregister_viewer(vid)
                    overlay_hub.unsubscribe(vid)
                break

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("ws_overlays: connection error viewer=%s", viewer_id)
    finally:
        if push_task and not push_task.done():
            push_task.cancel()
        if viewer_id:
            await detection_worker.unregister_viewer(viewer_id)
            overlay_hub.unsubscribe(viewer_id)
        try:
            await websocket.close()
        except Exception:
            pass


async def _push_overlays(websocket: WebSocket, queue, viewer_id: str) -> None:
    try:
        while True:
            payload = await queue.get()
            if payload is None:
                break
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception:
                break
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("_push_overlays: stopped for viewer=%s", viewer_id)


# --- Static frontend serving ---

_dist = FRONTEND_DIST_DIR
if os.path.isdir(_dist):
    _assets_dir = os.path.join(_dist, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(os.path.join(_dist, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        candidate = os.path.join(_dist, full_path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_dist, "index.html"))
