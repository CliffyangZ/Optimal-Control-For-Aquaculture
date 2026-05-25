"""FastAPI application for the RF-DETR live monitoring web UI."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from config import setting
from core.models import DetectionHealthStatus, StreamStatus
from web.detection_worker import worker
from web.overlay_hub import overlay_hub
from web.schemas import ConfigResponse, HeartbeatRequest, SessionStatusResponse, SessionStopRequest


LOOPBACK_MEDIAMTX_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}

app = FastAPI(title="Aquaculture RF-DETR Monitor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _request_host(request: Request) -> str:
    host = request.headers.get("host", "localhost")
    return host.split(":", 1)[0]


def _webrtc_host(request: Request) -> str:
    if setting.MEDIAMTX_HOST in LOOPBACK_MEDIAMTX_HOSTS:
        return _request_host(request)
    return setting.MEDIAMTX_HOST


def _scheme_for_request(request: Request, ws: bool = False) -> str:
    forwarded_proto = request.headers.get("x-forwarded-proto")
    https = forwarded_proto == "https" or request.url.scheme == "https"
    if ws:
        return "wss" if https else "ws"
    return "https" if https else "http"


def _build_webrtc_url(request: Request) -> str:
    path = setting.WEBRTC_PATH_TEMPLATE.format(source_id=setting.SOURCE_ID)
    return f"{_scheme_for_request(request)}://{_webrtc_host(request)}:{setting.MEDIAMTX_WEBRTC_PORT}{path}"


def _build_overlay_ws_url(request: Request) -> str:
    return f"{_scheme_for_request(request, ws=True)}://{request.headers.get('host', 'localhost')}/ws/overlays"


def _status_response(viewer_id: str | None = None) -> SessionStatusResponse:
    session = worker.get_session_snapshot()
    if session is None:
        return SessionStatusResponse(
            source_id=setting.SOURCE_ID,
            stream_status=StreamStatus.WAITING,
            detection_status=DetectionHealthStatus.UNAVAILABLE,
            active_viewer_count=0,
            message=worker.status_message,
            viewer_id=viewer_id,
        )
    return SessionStatusResponse(
        source_id=session.source_id,
        stream_status=session.stream_status,
        detection_status=session.detection_status,
        active_viewer_count=session.active_viewer_count,
        detection_fps=session.detection_fps,
        last_frame_at=session.last_frame_at.isoformat() if session.last_frame_at else None,
        last_overlay_at=session.last_overlay_at.isoformat() if session.last_overlay_at else None,
        message=worker.status_message,
        viewer_id=viewer_id,
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config(request: Request) -> ConfigResponse:
    return ConfigResponse(
        source_id=setting.SOURCE_ID,
        webrtc_url=_build_webrtc_url(request),
        overlay_ws_url=_build_overlay_ws_url(request),
        stale_overlay_ms=setting.STALE_OVERLAY_MS,
        local_trusted_network_only=setting.LOCAL_TRUSTED_NETWORK_ONLY,
        detector_model_configured=bool(setting.RFDETR_MODEL_PATH),
        confidence_threshold=setting.RFDETR_CONFIDENCE_THRESHOLD,
        device=worker.device,
    )


@app.get("/status", response_model=SessionStatusResponse)
async def get_status() -> SessionStatusResponse:
    return _status_response()


@app.post("/session/start", response_model=SessionStatusResponse)
async def session_start() -> SessionStatusResponse:
    viewer_id = uuid4().hex
    await worker.register_viewer(viewer_id)
    response = _status_response(viewer_id=viewer_id)
    response.message = f"viewer_id:{viewer_id}"
    return response


@app.post("/session/heartbeat", response_model=SessionStatusResponse)
async def session_heartbeat(payload: HeartbeatRequest) -> SessionStatusResponse:
    await worker.update_heartbeat(payload.viewer_id)
    return _status_response(viewer_id=payload.viewer_id)


@app.post("/session/stop", response_model=SessionStatusResponse)
async def session_stop(payload: SessionStopRequest | None = None) -> SessionStatusResponse:
    viewer_id = payload.viewer_id if payload else None
    await worker.unregister_viewer(viewer_id)
    return _status_response(viewer_id=viewer_id)


@app.websocket("/ws/overlays")
async def websocket_overlays(websocket: WebSocket) -> None:
    await websocket.accept()
    viewer_id = websocket.query_params.get("viewer_id") or uuid4().hex
    queue = overlay_hub.subscribe(viewer_id)
    try:
        while True:
            payload = await queue.get()
            if payload is None:
                break
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        overlay_hub.unsubscribe(viewer_id)


frontend_dist = Path(setting.FRONTEND_DIST_DIR)
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", response_model=None)
    async def serve_frontend(full_path: str) -> Response:
        index = frontend_dist / "index.html"
        if index.exists():
            return FileResponse(index)
        return JSONResponse({"message": "Frontend build is not available"}, status_code=404)
