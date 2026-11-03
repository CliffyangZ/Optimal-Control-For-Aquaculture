"""Async detection worker that manages a shared inference loop.

Handles:
- Viewer registration/unregistration (starts worker on first viewer, stops on last)
- Background frame reading via LatestFrameReader
- RF-DETR inference via RFDETRDetector
- WebSocket broadcast via overlay_hub
- FeedAnalysis for hunger level and feed area analytics
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.setting import (
    RFDETR_DEVICE,
    RFDETR_MODEL_PATH,
    RFDETR_CONFIDENCE_THRESHOLD,
    SOURCE_ID,
    STALE_OVERLAY_MS,
    STREAM_URL,
    TARGET_DETECTION_FPS,
    VIEWER_HEARTBEAT_TIMEOUT_SECONDS,
)
from core.infer.feed_analyzer import FeedAnalysis
from core.infer.rfdetr_detector import RFDETRDetector
from core.models import (
    DetectionHealthStatus,
    LiveStreamSession,
    StreamStatus,
    VideoFrame,
)
from core.stream_health import derive_detection_status, derive_stream_status
from core.types import Detection, DetectionResult
from streaming.latest_frame_reader import LatestFrameReader
from web.overlay_hub import overlay_hub
from web.schemas import AnalyticsMessage, DetectionPayload, DetectionsMessage


class DetectionWorker:
    """Manages a single shared LiveStreamSession with background inference loop."""

    def __init__(self) -> None:
        self._session: Optional[LiveStreamSession] = None
        self._viewer_ids: set[str] = set()
        self._worker_task: Optional[asyncio.Task] = None

        self._detector: Optional[RFDETRDetector] = None
        self._frame_reader: LatestFrameReader = LatestFrameReader(STREAM_URL)
        self._frame_reader.start()

        self._feed_analysis: FeedAnalysis = FeedAnalysis()
        self.status_message: str = ""

        self._frame_counter: int = 0
        self._overlay_intervals: list[float] = []
        self._last_overlay_monotonic: Optional[float] = None
        self._last_processed_frame_id: Optional[int] = None

        self._analytics_t0: float = time.monotonic()
        self._ANALYTICS_INTERVAL = 2.0

        self._load_detector()

    def _load_detector(self) -> None:
        """Load RF-DETR model or set unavailable status."""
        self.status_message = ""
        model_path_str = RFDETR_MODEL_PATH
        if not model_path_str or not model_path_str.strip():
            self._detector = None
            if self._session:
                self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
            self.status_message = "RF-DETR model unavailable"
            return

        model_path = Path(model_path_str)
        if not model_path.exists():
            self._detector = None
            if self._session:
                self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
            self.status_message = "RF-DETR model unavailable"
            return

        try:
            self._detector = RFDETRDetector.load(
                model_path,
                device=RFDETR_DEVICE,
                conf_threshold=RFDETR_CONFIDENCE_THRESHOLD,
            )
            if self._session:
                self._session.detection_status = DetectionHealthStatus.LOADING
        except Exception:
            self._detector = None
            if self._session:
                self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
            self.status_message = "RF-DETR model unavailable"

    async def register_viewer(self, viewer_id: Optional[str] = None) -> LiveStreamSession:
        """Register a new viewer and start the worker if first viewer."""
        if viewer_id is None:
            viewer_id = str(uuid.uuid4())

        if self._session is None:
            self._session = LiveStreamSession(
                session_id=str(uuid.uuid4()),
                source_id=SOURCE_ID,
                started_at=datetime.now(timezone.utc),
            )
            self._reset_runtime_metrics()

        if viewer_id not in self._viewer_ids:
            self._viewer_ids.add(viewer_id)
            self._session.active_viewer_count = len(self._viewer_ids)

        if self._worker_task is None or self._worker_task.done():
            await self._start_worker()

        return self._session

    async def _start_worker(self) -> None:
        """Start the background inference worker task."""
        self._worker_task = asyncio.create_task(self._run_loop())

    async def unregister_viewer(self, viewer_id: str) -> None:
        """Unregister a viewer (safe for unknown viewer_id)."""
        if viewer_id in self._viewer_ids:
            self._viewer_ids.discard(viewer_id)
            if self._session:
                self._session.active_viewer_count = len(self._viewer_ids)

    async def update_heartbeat(self, viewer_id: str) -> None:
        """Update heartbeat for a viewer (safe for unknown viewer_id)."""
        pass

    def get_session_snapshot(self) -> Optional[LiveStreamSession]:
        """Get current session snapshot or None if no viewers registered yet."""
        return self._session

    def _validated_detections(self, detections: list[Detection]) -> list[Detection]:
        """Return all detections from model (no filtering)."""
        return detections

    def _sync_detection_state_for_frame_source(self, has_frame: bool) -> None:
        """Set detection status based on frame availability."""
        if not has_frame and self._session:
            self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
            self.status_message = "Detection stream unavailable"

    def _should_process_frame(self, frame_id: int) -> bool:
        """Deduplication: return True for new frame_id, False for duplicate."""
        if self._last_processed_frame_id != frame_id:
            self._last_processed_frame_id = frame_id
            return True
        return False

    def _refresh_detection_state(self, now: datetime) -> None:
        """Check staleness and update detection_status."""
        if not self._session:
            return

        if self._session.last_overlay_at is None:
            return

        age_ms = (now - self._session.last_overlay_at).total_seconds() * 1000
        if age_ms > STALE_OVERLAY_MS:
            self._session.detection_status = DetectionHealthStatus.STALE

    def _record_overlay_timing(self, fps: float) -> None:
        """Record overlay timing for FPS calculation."""
        if fps > 0:
            interval = 1.0 / fps
            self._overlay_intervals.append(interval)
            if len(self._overlay_intervals) > 30:
                self._overlay_intervals.pop(0)

        if self._overlay_intervals and self._session:
            avg_interval = sum(self._overlay_intervals) / len(self._overlay_intervals)
            if avg_interval > 0:
                self._session.detection_fps = 1.0 / avg_interval

        self._last_overlay_monotonic = time.monotonic()

    def _reset_runtime_metrics(self) -> None:
        """Reset runtime metrics."""
        self._frame_counter = 0
        self._overlay_intervals = []
        self._last_overlay_monotonic = None
        self._last_processed_frame_id = None
        self._feed_analysis.reset()
        self._analytics_t0 = time.monotonic()

    async def _broadcast_overlay(
        self, video_frame: VideoFrame, result: DetectionResult, now_ms: int
    ) -> None:
        """Build and broadcast detection overlay message."""
        if not self._session:
            return

        detection_payloads = []
        for det in result.detections:
            payload = DetectionPayload(
                class_name=det.class_name,
                class_index=det.class_index,
                confidence=det.confidence,
                x=det.bbox_x,
                y=det.bbox_y,
                width=det.bbox_w,
                height=det.bbox_h,
            )
            detection_payloads.append(payload)

        msg = DetectionsMessage(
            source_id=self._session.source_id,
            frame_id=result.frame_id,
            timestamp_ms=now_ms,
            source_width=result.source_width,
            source_height=result.source_height,
            stream_status=self._session.stream_status,
            detection_status=self._session.detection_status,
            detections=detection_payloads,
        )

        overlay_hub.broadcast(msg.to_ws_dict())

    def _maybe_broadcast_analytics(self, now_ms: int) -> None:
        """Broadcast analytics if interval elapsed."""
        if not self._session:
            return

        now = time.monotonic()
        if now - self._analytics_t0 < self._ANALYTICS_INTERVAL:
            return

        self._analytics_t0 = now

        chart_data = self._feed_analysis.get_chart_data()
        latest_snapshot = self._feed_analysis.get_latest_snapshot()

        if not latest_snapshot or not chart_data["frame_ids"]:
            return

        msg = AnalyticsMessage(
            source_id=self._session.source_id,
            timestamp_ms=now_ms,
            hunger_level=latest_snapshot.hunger_level,
            consumption_rate=latest_snapshot.consumption_rate,
            frame_ids=chart_data["frame_ids"],
            raw_area=chart_data["raw"],
            kalman_area=chart_data["kalman"],
            ema_area=chart_data["ema"],
        )

        overlay_hub.broadcast(msg.to_ws_dict())

    async def _run_loop(self) -> None:
        """Background inference loop."""
        if not self._session:
            return

        self._session.detection_status = DetectionHealthStatus.LOADING if self._detector else DetectionHealthStatus.UNAVAILABLE
        target_interval = 1.0 / TARGET_DETECTION_FPS if TARGET_DETECTION_FPS > 0 else 0.1

        try:
            while self._viewer_ids:
                t_loop_start = time.monotonic()
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                current_frame = self._frame_reader.get_latest_frame()
                if current_frame is None:
                    self._sync_detection_state_for_frame_source(False)
                    await asyncio.sleep(0.05)
                    continue

                self._sync_detection_state_for_frame_source(True)
                self._session.last_frame_at = current_frame.captured_at

                if not self._should_process_frame(current_frame.sequence_number):
                    await asyncio.sleep(0.01)
                    continue

                if self._detector is None:
                    self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
                    await asyncio.sleep(0.05)
                    continue

                self._frame_counter += 1

                try:
                    result = await asyncio.to_thread(
                        self._detector.detect,
                        current_frame.image,
                        current_frame.sequence_number,
                    )
                except Exception:
                    await asyncio.sleep(0.05)
                    continue

                result.detections = self._validated_detections(result.detections)

                snapshot = self._feed_analysis.analyze_frame(result)

                now = datetime.now(timezone.utc)
                self._refresh_detection_state(now)

                self._session.last_overlay_at = now
                if self._session.detection_status == DetectionHealthStatus.LOADING:
                    self._session.detection_status = DetectionHealthStatus.ACTIVE

                video_frame = VideoFrame(
                    frame_id=current_frame.sequence_number,
                    captured_at=current_frame.captured_at,
                    source_width=current_frame.source_width,
                    source_height=current_frame.source_height,
                    image=current_frame.image,
                )

                await self._broadcast_overlay(video_frame, result, now_ms)
                self._record_overlay_timing(0.0)

                self._maybe_broadcast_analytics(now_ms)

                elapsed = time.monotonic() - t_loop_start
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            if self._session:
                self._session.detection_status = DetectionHealthStatus.UNAVAILABLE
