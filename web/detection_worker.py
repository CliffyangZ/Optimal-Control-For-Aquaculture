"""Shared detection worker lifecycle for one configured stream."""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from config.setting import (
    SOURCE_ID,
    STREAM_URL,
    TARGET_DETECTION_FPS,
    VIEWER_HEARTBEAT_TIMEOUT_SECONDS,
    IDLE_SHUTDOWN_DELAY_SECONDS,
    STALE_OVERLAY_MS,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_DEVICE,
    YOLO_MODEL_PATH,
    YOLO_SUPPORTED_CLASSES,
)
from core.models import (
    DetectionHealthStatus,
    LiveStreamSession,
    StreamStatus,
    VideoFrame,
    ViewerSession,
    ViewerState,
)
from core.stream_health import derive_detection_status, derive_stream_status
from core.types import Detection, DetectionResult
from streaming.latest_frame_reader import LatestFrameReader
from web.overlay_hub import overlay_hub
from web.schemas import DetectionsMessage, DetectionPayload, StatusMessage

logger = logging.getLogger(__name__)


class DetectionWorker:
    """Single shared detection loop for the configured stream.

    Lifecycle:
    - Starts when the first viewer joins via register_viewer().
    - Runs detection at TARGET_DETECTION_FPS and broadcasts detections.v1 over overlay_hub.
    - Stops within IDLE_SHUTDOWN_DELAY_SECONDS after active_viewer_count reaches 0.
    """

    def __init__(self) -> None:
        self._session: Optional[LiveStreamSession] = None
        self._viewers: dict[str, ViewerSession] = {}
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._frame_reader: Optional[LatestFrameReader] = None
        self._detector = None  # YOLODetector; loaded lazily in _start_worker
        self._frame_counter = 0
        self._detection_times: list[float] = []
        self._status_message: Optional[str] = None

    # --- Public API ---

    @property
    def session(self) -> Optional[LiveStreamSession]:
        return self._session

    async def register_viewer(self, viewer_id: str, overlay_enabled: bool = True) -> LiveStreamSession:
        async with self._lock:
            now = datetime.now(timezone.utc)
            viewer = ViewerSession(
                viewer_id=viewer_id,
                connected_at=now,
                last_heartbeat_at=now,
                state=ViewerState.ACTIVE,
                overlay_enabled=overlay_enabled,
            )
            self._viewers[viewer_id] = viewer

            if self._session is None:
                self._session = LiveStreamSession(
                    session_id=str(uuid4()),
                    source_id=SOURCE_ID,
                    started_at=now,
                    stream_status=StreamStatus.CONNECTING,
                    detection_status=DetectionHealthStatus.LOADING,
                )
                self._start_worker()
            self._session.active_viewer_count = len(self._active_viewers())
            return self._session

    async def unregister_viewer(self, viewer_id: str) -> Optional[LiveStreamSession]:
        async with self._lock:
            viewer = self._viewers.pop(viewer_id, None)
            if viewer:
                viewer.state = ViewerState.STOPPED
            if self._session:
                self._session.active_viewer_count = len(self._active_viewers())
            return self._session

    async def update_heartbeat(self, viewer_id: str) -> None:
        async with self._lock:
            viewer = self._viewers.get(viewer_id)
            if viewer:
                viewer.last_heartbeat_at = datetime.now(timezone.utc)

    def get_session_snapshot(self) -> Optional[LiveStreamSession]:
        return self._session

    @property
    def status_message(self) -> Optional[str]:
        return self._status_message

    # --- Internal helpers ---

    def _active_viewers(self) -> list[ViewerSession]:
        return [v for v in self._viewers.values() if v.state in (ViewerState.ACTIVE, ViewerState.CONNECTING, ViewerState.RECONNECTING)]

    def _start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._frame_reader = LatestFrameReader(STREAM_URL)
            self._frame_reader.start()
            self._load_detector()
            self._worker_task = asyncio.create_task(self._run())

    def _configured_model_path(self) -> str:
        return os.environ.get("YOLO_MODEL_PATH", YOLO_MODEL_PATH).strip()

    def _configured_device(self) -> str:
        return os.environ.get("YOLO_DEVICE", YOLO_DEVICE).strip() or "cpu"

    def _set_detection_state(self, status: DetectionHealthStatus, message: Optional[str] = None) -> None:
        self._status_message = message
        if self._session:
            self._session.detection_status = status

    def _load_detector(self) -> None:
        if self._detector is not None:
            self._set_detection_state(DetectionHealthStatus.ACTIVE)
            return

        model_path = self._configured_model_path()
        if not model_path:
            self._set_detection_state(
                DetectionHealthStatus.UNAVAILABLE,
                "Detection model unavailable",
            )
            logger.warning("detection_worker: YOLO_MODEL_PATH is not configured; no fallback detector will be used")
            return

        self._set_detection_state(DetectionHealthStatus.LOADING, "Loading detection model")
        try:
            from core.infer.detector import YOLODetector

            device = self._configured_device()
            self._detector = YOLODetector.load(
                model_path,
                device=device,
                conf_threshold=YOLO_CONFIDENCE_THRESHOLD,
            )
            self._set_detection_state(DetectionHealthStatus.ACTIVE)
            logger.info("detection_worker: loaded approved YOLODetector on %s", device)
        except Exception:
            self._detector = None
            self._set_detection_state(
                DetectionHealthStatus.UNAVAILABLE,
                "Detection model unavailable",
            )
            logger.exception("detection_worker: failed to load configured YOLODetector")

    async def _run(self) -> None:
        logger.info("detection_worker: started for source=%s", SOURCE_ID)
        detection_interval = 1.0 / TARGET_DETECTION_FPS if TARGET_DETECTION_FPS > 0 else 0.1
        idle_since: Optional[float] = None

        try:
            while True:
                loop_start = time.monotonic()

                # Check if we should idle-shutdown
                active = self._active_viewers()
                if not active:
                    if idle_since is None:
                        idle_since = time.monotonic()
                    elif time.monotonic() - idle_since >= IDLE_SHUTDOWN_DELAY_SECONDS:
                        logger.info("detection_worker: no active viewers; shutting down")
                        break
                else:
                    idle_since = None

                # Heartbeat timeout cleanup
                now = datetime.now(timezone.utc)
                async with self._lock:
                    for viewer_id, v in list(self._viewers.items()):
                        age = (now - v.last_heartbeat_at).total_seconds()
                        if age > VIEWER_HEARTBEAT_TIMEOUT_SECONDS:
                            logger.info("detection_worker: heartbeat timeout viewer=%s", viewer_id)
                            v.state = ViewerState.TIMED_OUT
                            self._viewers.pop(viewer_id, None)
                    if self._session:
                        self._session.active_viewer_count = len(self._active_viewers())

                # Get latest frame from reader
                if self._frame_reader is None:
                    await asyncio.sleep(detection_interval)
                    continue

                raw_frame = self._frame_reader.get_latest_frame()
                last_frame_time = self._frame_reader.get_last_frame_time()

                if self._session:
                    if last_frame_time:
                        self._session.last_frame_at = last_frame_time
                    self._session.stream_status = derive_stream_status(
                        last_frame_at=last_frame_time,
                        started_at=self._session.started_at,
                    )

                if raw_frame is None:
                    await self._broadcast_status()
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0, detection_interval - elapsed))
                    continue

                self._frame_counter += 1
                video_frame = VideoFrame(
                    frame_id=raw_frame.sequence_number,
                    captured_at=raw_frame.captured_at,
                    source_width=raw_frame.source_width,
                    source_height=raw_frame.source_height,
                    image=raw_frame.image,
                )

                if self._detector is None:
                    await self._broadcast_status()
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0, detection_interval - elapsed))
                    continue

                # Run YOLO inference (offload to thread executor to avoid blocking event loop)
                loop = asyncio.get_event_loop()
                t0 = time.monotonic()
                result: DetectionResult = await loop.run_in_executor(
                    None, self._detector.detect, raw_frame.image
                )
                result.frame_id = video_frame.frame_id
                result.captured_at = video_frame.captured_at
                result.detections = self._validated_detections(result.detections)
                if self._session and self._session.detection_status == DetectionHealthStatus.UNAVAILABLE:
                    await self._broadcast_status()
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0, detection_interval - elapsed))
                    continue
                detection_time = time.monotonic() - t0
                self._detection_times.append(detection_time)
                if len(self._detection_times) > 30:
                    self._detection_times.pop(0)

                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                if self._session:
                    self._session.last_overlay_at = datetime.now(timezone.utc)
                    avg_dt = sum(self._detection_times) / len(self._detection_times) if self._detection_times else 0
                    self._session.detection_fps = round(1.0 / avg_dt, 1) if avg_dt > 0 else 0.0
                    self._session.detection_status = derive_detection_status(
                        last_overlay_at=self._session.last_overlay_at,
                        last_frame_at=last_frame_time,
                        stale_seconds=STALE_OVERLAY_MS / 1000,
                    )

                await self._broadcast_overlay(video_frame, result, now_ms)

                elapsed = time.monotonic() - loop_start
                await asyncio.sleep(max(0, detection_interval - elapsed))

        except asyncio.CancelledError:
            logger.info("detection_worker: cancelled")
        except Exception:
            logger.exception("detection_worker: unhandled error")
        finally:
            await self._shutdown()

    async def _broadcast_overlay(self, frame: VideoFrame, result: DetectionResult, now_ms: int) -> None:
        if self._session is None:
            return

        detection_payloads = [
            DetectionPayload(
                class_name=d.class_name,
                confidence=d.confidence,
                x=d.bbox_x,
                y=d.bbox_y,
                width=d.bbox_w,
                height=d.bbox_h,
            )
            for d in result.detections
        ]

        msg = DetectionsMessage(
            schema="detections.v1",
            source_id=self._session.source_id,
            frame_id=frame.frame_id,
            timestamp_ms=now_ms,
            source_width=frame.source_width,
            source_height=frame.source_height,
            stream_status=self._session.stream_status,
            detection_status=self._session.detection_status,
            detections=detection_payloads,
        )
        overlay_hub.broadcast(msg.to_ws_dict())

    def _validated_detections(self, detections: list[Detection]) -> list[Detection]:
        valid: list[Detection] = []
        supported = set(YOLO_SUPPORTED_CLASSES)
        for detection in detections:
            if detection.class_name not in supported:
                self._set_detection_state(
                    DetectionHealthStatus.UNAVAILABLE,
                    "Detection model class mismatch",
                )
                logger.error("detection_worker: unsupported YOLO class %s", detection.class_name)
                return []
            if detection.confidence >= YOLO_CONFIDENCE_THRESHOLD:
                valid.append(detection)
        return valid

    async def _broadcast_status(self) -> None:
        if self._session is None:
            return
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        msg = StatusMessage(
            schema="status.v1",
            source_id=self._session.source_id,
            timestamp_ms=now_ms,
            stream_status=self._session.stream_status,
            detection_status=self._session.detection_status,
            active_viewer_count=self._session.active_viewer_count,
            message=self._status_message,
        )
        overlay_hub.broadcast(msg.to_ws_dict())

    async def _shutdown(self) -> None:
        if self._frame_reader:
            try:
                self._frame_reader.stop()
            except Exception:
                pass
            self._frame_reader = None
        overlay_hub.close_all()
        async with self._lock:
            if self._session:
                self._session.stopped_at = datetime.now(timezone.utc)
                self._session.stream_status = StreamStatus.WAITING
                self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
                self._session.active_viewer_count = 0
            self._session = None
            self._viewers.clear()
        logger.info("detection_worker: shutdown complete")


detection_worker = DetectionWorker()
