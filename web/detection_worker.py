"""Viewer-driven RF-DETR detection worker for live monitoring."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from config import setting
from core.infer.rfdetr_detector import RFDETRDetector
from core.models import CurrentFrame, DetectionHealthStatus, LiveStreamSession, StreamStatus, ViewerSession, ViewerState, VideoFrame
from core.stream_health import derive_detection_status, derive_stream_status
from core.types import Detection, DetectionResult
from streaming.latest_frame_reader import LatestFrameReader
from web.overlay_hub import overlay_hub
from web.schemas import DetectionPayload, DetectionsMessage, StatusMessage


class DetectionWorker:
    def __init__(self) -> None:
        self.source_id = setting.SOURCE_ID
        self.status_message = "Waiting for viewer"
        self.model_error: str | None = None
        self.device = self._resolve_device(setting.RFDETR_DEVICE)
        self.confidence_threshold = setting.RFDETR_CONFIDENCE_THRESHOLD

        self._session: LiveStreamSession | None = None
        self._viewers: dict[str, ViewerSession] = {}
        self._worker_task: asyncio.Task | None = None
        self._shutdown_task: asyncio.Task | None = None
        self._frame_reader: LatestFrameReader | None = None
        self._detector: RFDETRDetector | None = None

        self._lock = asyncio.Lock()
        self._last_processed_frame_id: int | None = None
        self._last_overlay_monotonic: float | None = None
        self._overlay_intervals: list[float] = []
        self._frame_counter = 0

    async def register_viewer(self, viewer_id: str | None = None) -> LiveStreamSession:
        viewer_id = viewer_id or uuid4().hex
        now = datetime.now(timezone.utc)
        async with self._lock:
            if self._shutdown_task and not self._shutdown_task.done():
                self._shutdown_task.cancel()
                self._shutdown_task = None
            if self._session is None:
                self._session = LiveStreamSession(session_id=uuid4().hex, source_id=self.source_id, started_at=now)
            self._viewers[viewer_id] = ViewerSession(
                viewer_id=viewer_id,
                connected_at=now,
                last_heartbeat_at=now,
                state=ViewerState.ACTIVE,
            )
            self._session.active_viewer_count = len(self._viewers)
            self._start_worker()
            return self._session

    async def unregister_viewer(self, viewer_id: str | None) -> LiveStreamSession | None:
        async with self._lock:
            if viewer_id:
                self._viewers.pop(viewer_id, None)
            elif self._viewers:
                self._viewers.pop(next(iter(self._viewers)))
            if self._session is None:
                return None
            self._session.active_viewer_count = len(self._viewers)
            if not self._viewers and (self._shutdown_task is None or self._shutdown_task.done()):
                self._shutdown_task = asyncio.create_task(self._stop_after_idle_delay())
            return self._session

    async def update_heartbeat(self, viewer_id: str) -> LiveStreamSession | None:
        async with self._lock:
            viewer = self._viewers.get(viewer_id)
            if viewer:
                viewer.last_heartbeat_at = datetime.now(timezone.utc)
                viewer.state = ViewerState.ACTIVE
            return self._session

    def get_session_snapshot(self) -> LiveStreamSession | None:
        return self._session

    def _start_worker(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._reset_runtime_metrics()
        self._worker_task = asyncio.create_task(self._run_loop())

    async def _stop_after_idle_delay(self) -> None:
        await asyncio.sleep(setting.IDLE_SHUTDOWN_DELAY_SECONDS)
        async with self._lock:
            if self._viewers:
                return
            if self._worker_task and not self._worker_task.done():
                self._worker_task.cancel()
            self._stop_frame_reader()
            if self._session:
                self._session.stopped_at = datetime.now(timezone.utc)
                self._session.stream_status = StreamStatus.WAITING
                self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
            self.status_message = "Waiting for viewer"

    async def _run_loop(self) -> None:
        self._load_detector()
        if self._detector is None:
            self._broadcast_status()
            return

        self._frame_reader = LatestFrameReader(setting.STREAM_URL)
        self._frame_reader.start()
        try:
            while self._session is not None and self._session.active_viewer_count > 0:
                await self._process_latest_frame_once()
                await asyncio.sleep(max(0.05, 1.0 / max(1, setting.TARGET_DETECTION_FPS)))
        except asyncio.CancelledError:
            raise
        finally:
            self._stop_frame_reader()

    async def _process_latest_frame_once(self) -> None:
        if self._session is None or self._frame_reader is None or self._detector is None:
            return

        now = datetime.now(timezone.utc)
        latest_frame = self._frame_reader.get_latest_frame()
        self._session.last_frame_at = self._frame_reader.get_last_frame_time()
        self._session.stream_status = derive_stream_status(
            self._session.last_frame_at,
            self._session.started_at,
            setting.INTERRUPTION_TIMEOUT_SECONDS,
        )

        if latest_frame is None:
            self._sync_detection_state_for_frame_source(False)
            self._broadcast_status()
            return
        self._sync_detection_state_for_frame_source(True)
        if not self._should_process_frame(latest_frame.sequence_number):
            self._refresh_detection_state(now)
            return

        video_frame = VideoFrame(
            frame_id=latest_frame.sequence_number,
            captured_at=latest_frame.captured_at,
            source_width=latest_frame.source_width,
            source_height=latest_frame.source_height,
            image=latest_frame.image,
        )
        result = self._detector.detect(latest_frame.image, frame_id=latest_frame.sequence_number)
        await self._broadcast_overlay(video_frame, result, int(now.timestamp() * 1000))

    def _load_detector(self) -> None:
        model_path = setting.RFDETR_MODEL_PATH
        if not model_path or not Path(model_path).exists():
            self._detector = None
            self.model_error = "RF-DETR model unavailable"
            self.status_message = "RF-DETR model unavailable"
            if self._session:
                self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
            return
        try:
            self._detector = RFDETRDetector.load(model_path, device=self.device, conf_threshold=self.confidence_threshold)
            self.model_error = None
            self.status_message = "Detection active"
            if self._session:
                self._session.detection_status = DetectionHealthStatus.LOADING
        except Exception:
            self._detector = None
            self.model_error = "RF-DETR model unavailable"
            self.status_message = "RF-DETR model unavailable"
            if self._session:
                self._session.detection_status = DetectionHealthStatus.UNAVAILABLE

    async def _broadcast_overlay(self, video_frame: VideoFrame, result: DetectionResult, now_ms: int | None = None) -> None:
        if self._session is None:
            return
        now_ms = now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
        self._session.last_overlay_at = datetime.now(timezone.utc)
        self._session.detection_status = DetectionHealthStatus.ACTIVE
        self._record_overlay_timing(time.monotonic())
        detections = [self._to_payload(detection) for detection in self._validated_detections(result.detections)]
        msg = DetectionsMessage(
            source_id=self.source_id,
            frame_id=video_frame.frame_id,
            timestamp_ms=now_ms,
            source_width=video_frame.source_width,
            source_height=video_frame.source_height,
            stream_status=self._session.stream_status,
            detection_status=self._session.detection_status,
            detections=detections,
            inference_ms=result.inference_ms,
            detection_fps=self._session.detection_fps,
            message=self.status_message,
        )
        overlay_hub.broadcast(msg.to_ws_dict())

    def _broadcast_status(self) -> None:
        if self._session is None:
            return
        msg = StatusMessage(
            source_id=self.source_id,
            timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            stream_status=self._session.stream_status,
            detection_status=self._session.detection_status,
            active_viewer_count=self._session.active_viewer_count,
            detection_fps=self._session.detection_fps,
            message=self.status_message,
        )
        overlay_hub.broadcast(msg.to_ws_dict())

    def _validated_detections(self, detections: list[Detection]) -> list[Detection]:
        return detections

    def _to_payload(self, detection: Detection) -> DetectionPayload:
        return DetectionPayload(
            class_name=detection.class_name,
            class_index=detection.class_index,
            confidence=detection.confidence,
            x=detection.bbox_x,
            y=detection.bbox_y,
            width=detection.bbox_w,
            height=detection.bbox_h,
        )

    def _sync_detection_state_for_frame_source(self, has_frame: bool) -> None:
        if self._session is None:
            return
        if not has_frame and self._session.stream_status == StreamStatus.STREAM_UNAVAILABLE:
            self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
            self.status_message = "Detection stream unavailable"
        elif has_frame and self._detector is not None and self._session.last_overlay_at is None:
            self._session.detection_status = DetectionHealthStatus.LOADING
            self.status_message = "Detection loading"

    def _should_process_frame(self, frame_id: int) -> bool:
        if self._last_processed_frame_id == frame_id:
            return False
        self._last_processed_frame_id = frame_id
        return True

    def _refresh_detection_state(self, now: datetime | None = None) -> None:
        if self._session is None:
            return
        self._session.detection_status = derive_detection_status(
            self._session.last_overlay_at,
            self._session.last_frame_at,
            stale_seconds=setting.STALE_OVERLAY_MS / 1000,
            detector_available=self._detector is not None,
        )

    def _record_overlay_timing(self, now_monotonic: float) -> None:
        if self._session is None:
            return
        if self._last_overlay_monotonic is not None:
            interval = max(0.0001, now_monotonic - self._last_overlay_monotonic)
            self._overlay_intervals.append(interval)
            if len(self._overlay_intervals) > 30:
                self._overlay_intervals.pop(0)
            avg = sum(self._overlay_intervals) / len(self._overlay_intervals)
            self._session.detection_fps = round(1.0 / avg, 1)
        self._last_overlay_monotonic = now_monotonic

    def _reset_runtime_metrics(self) -> None:
        self._frame_counter = 0
        self._overlay_intervals = []
        self._last_overlay_monotonic = None
        self._last_processed_frame_id = None

    def _stop_frame_reader(self) -> None:
        if self._frame_reader is not None:
            self._frame_reader.stop()
            self._frame_reader = None

    def _resolve_device(self, configured: str) -> str:
        if configured and configured.lower() != "auto":
            return configured
        return "cuda" if torch.cuda.is_available() else "cpu"


worker = DetectionWorker()
