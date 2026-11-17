"""FastAPI backend for RF-DETR WebRTC aquaculture monitoring.

Exposes:
- /config - WebRTC WHEP URL and overlay WebSocket URL
- /status - current session status
- /session/start - register viewer and start inference
- /session/stop - unregister viewer
- /ws/overlays - WebSocket stream for detections.v1 and analytics.v1 messages
- /assets and / - static file serving (if frontend/dist exists)
"""

from __future__ import annotations

import asyncio
import queue
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.setting import (
    BACKEND_HOST,
    BACKEND_PORT,
    FRONTEND_DIST_DIR,
    INTERRUPTION_TIMEOUT_SECONDS,
    LOCAL_TRUSTED_NETWORK_ONLY,
    MEDIAMTX_HOST,
    MEDIAMTX_WEBRTC_PORT,
    MEDIAMTX_HLS_PORT,
    SOURCE_ID,
    STALE_OVERLAY_MS,
    WEBRTC_PATH_TEMPLATE,
    HLS_PATH_TEMPLATE,
)
from core.models import StreamStatus
from core.stream_health import derive_stream_status
from web.detection_worker import DetectionWorker
from web.overlay_hub import overlay_hub
from web.schemas import ConfigResponse, SessionStatusResponse

app = FastAPI(title="RF-DETR Aquaculture Monitor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOOPBACK_MEDIAMTX_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

detection_worker = DetectionWorker()


def _build_webrtc_url(request: Request) -> str:
    """Build WHEP URL, rewriting loopback to request client host."""
    host = MEDIAMTX_HOST
    port = MEDIAMTX_WEBRTC_PORT

    if host in LOOPBACK_MEDIAMTX_HOSTS and request.client:
        host = request.client.host

    webrtc_path = WEBRTC_PATH_TEMPLATE.format(source_id=SOURCE_ID)
    return f"http://{host}:{port}{webrtc_path}"


def _build_hls_url(request: Request) -> str:
    """Build HLS URL, rewriting loopback to request client host."""
    host = MEDIAMTX_HOST
    port = MEDIAMTX_HLS_PORT

    if host in LOOPBACK_MEDIAMTX_HOSTS and request.client:
        host = request.client.host

    hls_path = HLS_PATH_TEMPLATE.format(source_id=SOURCE_ID)
    return f"http://{host}:{port}{hls_path}"


def _build_overlay_ws_url(request: Request) -> str:
    """Build overlay WebSocket URL, using request host."""
    if request.client:
        host = request.client.host
    else:
        host = request.url.hostname or "localhost"

    scheme = "wss" if request.url.scheme == "https" else "ws"
    return f"{scheme}://{host}/ws/overlays"


@app.get("/config", response_model=ConfigResponse)
async def get_config(request: Request) -> ConfigResponse:
    """Get WebRTC WHEP URL, HLS URL, and overlay WebSocket URL."""
    return ConfigResponse(
        source_id=SOURCE_ID,
        webrtc_url=_build_webrtc_url(request),
        hls_url=_build_hls_url(request),
        overlay_ws_url=_build_overlay_ws_url(request),
        stale_overlay_ms=STALE_OVERLAY_MS,
        local_trusted_network_only=LOCAL_TRUSTED_NETWORK_ONLY,
    )


@app.get("/status", response_model=SessionStatusResponse)
async def get_status() -> SessionStatusResponse:
    """Get current session status."""
    session = detection_worker.get_session_snapshot()
    if session is None:
        return SessionStatusResponse(
            source_id=SOURCE_ID,
            stream_status=StreamStatus.WAITING.value,
            detection_status="unavailable",
            active_viewer_count=0,
        )

    stream_status = derive_stream_status(
        session.last_frame_at,
        session.started_at,
        interruption_timeout_seconds=INTERRUPTION_TIMEOUT_SECONDS,
    )
    session.stream_status = stream_status

    return SessionStatusResponse(
        source_id=session.source_id,
        stream_status=session.stream_status.value,
        detection_status=session.detection_status.value,
        active_viewer_count=session.active_viewer_count,
    )


@app.post("/session/start", response_model=SessionStatusResponse)
async def session_start(request: Request) -> SessionStatusResponse:
    """Register a viewer and start inference if first."""
    viewer_id = None
    try:
        body = await request.json()
        viewer_id = body.get("viewer_id")
    except Exception:
        pass

    session = await detection_worker.register_viewer(viewer_id)

    stream_status = derive_stream_status(
        session.last_frame_at,
        session.started_at,
        interruption_timeout_seconds=INTERRUPTION_TIMEOUT_SECONDS,
    )
    session.stream_status = stream_status

    return SessionStatusResponse(
        source_id=session.source_id,
        stream_status=session.stream_status.value,
        detection_status=session.detection_status.value,
        active_viewer_count=session.active_viewer_count,
    )


@app.post("/session/stop", response_model=SessionStatusResponse)
async def session_stop(request: Request) -> SessionStatusResponse:
    """Unregister a viewer."""
    viewer_id = None
    try:
        body = await request.json()
        viewer_id = body.get("viewer_id")
    except Exception:
        pass

    if viewer_id:
        await detection_worker.unregister_viewer(viewer_id)

    session = detection_worker.get_session_snapshot()
    if session is None:
        return SessionStatusResponse(
            source_id=SOURCE_ID,
            stream_status=StreamStatus.WAITING.value,
            detection_status="unavailable",
            active_viewer_count=0,
        )

    stream_status = derive_stream_status(
        session.last_frame_at,
        session.started_at,
        interruption_timeout_seconds=INTERRUPTION_TIMEOUT_SECONDS,
    )
    session.stream_status = stream_status

    return SessionStatusResponse(
        source_id=session.source_id,
        stream_status=session.stream_status.value,
        detection_status=session.detection_status.value,
        active_viewer_count=session.active_viewer_count,
    )


@app.websocket("/ws/overlays")
async def ws_overlays(websocket: WebSocket) -> None:
    """WebSocket endpoint for overlay messages (detections.v1, analytics.v1, status.v1)."""
    await websocket.accept()
    q = overlay_hub.subscribe("ws_client")

    try:
        while True:
            try:
                msg = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            if msg is None:
                break

            try:
                await websocket.send_json(msg)
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        overlay_hub.unsubscribe("ws_client")


dist_path = Path(FRONTEND_DIST_DIR)
if dist_path.exists() and dist_path.is_dir():
    app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        """Serve index.html for SPA routing."""
        index_path = dist_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return FileResponse(index_path, status_code=404)
