"""Shared detection worker lifecycle for one configured stream."""

import asyncio
import functools
import logging
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from core.infer.feed_analyzer import FeedAreaSnapshot

from config.setting import (
    SOURCE_ID,
    STREAM_URL,
    TARGET_DETECTION_FPS,
    VIEWER_HEARTBEAT_TIMEOUT_SECONDS,
    IDLE_SHUTDOWN_DELAY_SECONDS,
    STALE_OVERLAY_MS,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_DEVICE,
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
from core.infer.feed_analyzer import FeedAnalysis, FeedAreaSnapshot
from streaming.latest_frame_reader import LatestFrameReader
from web.overlay_hub import overlay_hub
from web.schemas import (
    DetectionsMessage,
    DetectionPayload,
    DetectorResult,
    FeedAreaData,
    FeedAnalysisHistory,
    StatusMessage,
)

logger = logging.getLogger("uvicorn.error")

DETECTION_STREAM_UNAVAILABLE_MESSAGE = "Detection stream unavailable"


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
        self._yolo_detector = None  # YOLODetector; loaded lazily in _start_worker
        self._rfdetr_detector = None  # RFDETRDetector; loaded lazily if configured
        self._frame_counter = 0
        self._overlay_intervals: list[float] = []
        self._last_overlay_monotonic: Optional[float] = None
        self._last_processed_frame_id: Optional[int] = None
        self._status_message: Optional[str] = None
        self._feed_analyzer = FeedAnalysis()  # Real-time feed area analysis

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
            self._reset_runtime_metrics()
            self._frame_reader = LatestFrameReader(STREAM_URL)
            self._frame_reader.start()
            self._worker_task = asyncio.create_task(self._run())
            # _load_detector() is invoked at the top of _run() via run_in_executor
            # so model loading never blocks the event loop

    def _reset_runtime_metrics(self) -> None:
        self._frame_counter = 0
        self._overlay_intervals.clear()
        self._last_overlay_monotonic = None
        self._last_processed_frame_id = None

    def _configured_model_path(self) -> str:
        return os.environ.get("YOLO_MODEL_PATH", "").strip()

    def _configured_rfdetr_model_path(self) -> str:
        return os.environ.get("RFDETR_MODEL_PATH", "").strip()

    def _configured_device(self) -> str:
        return os.environ.get("YOLO_DEVICE", YOLO_DEVICE).strip() or "cpu"

    def _set_detection_state(self, status: DetectionHealthStatus, message: Optional[str] = None) -> None:
        self._status_message = message
        if self._session:
            self._session.detection_status = status

    def _sync_detection_state_for_frame_source(self, has_frame: bool) -> None:
        if self._session is None or (self._yolo_detector is None and self._rfdetr_detector is None):
            return
        if has_frame:
            if self._status_message == DETECTION_STREAM_UNAVAILABLE_MESSAGE:
                self._set_detection_state(DetectionHealthStatus.LOADING)
            return
        if self._session.stream_status == StreamStatus.STREAM_UNAVAILABLE:
            self._set_detection_state(
                DetectionHealthStatus.UNAVAILABLE,
                DETECTION_STREAM_UNAVAILABLE_MESSAGE,
            )

    def _refresh_detection_state(self, last_frame_at: Optional[datetime]) -> None:
        if self._session is None or (self._yolo_detector is None and self._rfdetr_detector is None):
            return
        if self._session.stream_status == StreamStatus.STREAM_UNAVAILABLE:
            self._set_detection_state(
                DetectionHealthStatus.UNAVAILABLE,
                DETECTION_STREAM_UNAVAILABLE_MESSAGE,
            )
            return
        if (
            self._session.detection_status == DetectionHealthStatus.UNAVAILABLE
            and self._status_message != DETECTION_STREAM_UNAVAILABLE_MESSAGE
        ):
            return
        self._session.detection_status = derive_detection_status(
            last_overlay_at=self._session.last_overlay_at,
            last_frame_at=last_frame_at,
            stale_seconds=STALE_OVERLAY_MS / 1000,
        )

    def _record_overlay_timing(self, overlay_monotonic: float) -> None:
        if self._session is None:
            return
        if self._last_overlay_monotonic is not None:
            interval = overlay_monotonic - self._last_overlay_monotonic
            if interval > 0:
                self._overlay_intervals.append(interval)
                if len(self._overlay_intervals) > 30:
                    self._overlay_intervals.pop(0)
        self._last_overlay_monotonic = overlay_monotonic

        if self._overlay_intervals:
            avg_interval = sum(self._overlay_intervals) / len(self._overlay_intervals)
            self._session.detection_fps = round(1.0 / avg_interval, 1) if avg_interval > 0 else 0.0

    def _should_process_frame(self, frame_id: int) -> bool:
        if self._last_processed_frame_id == frame_id:
            return False
        self._last_processed_frame_id = frame_id
        return True

    def _load_detector(self) -> None:
        if self._yolo_detector is not None:
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

        device = self._configured_device()
        self._set_detection_state(DetectionHealthStatus.LOADING, "Loading detection model")
        try:
            from core.infer.detector import YOLODetector

            logger.info(
                "detection_worker: loading YOLODetector model_path=%s device=%s conf_threshold=%.3f",
                model_path,
                device,
                YOLO_CONFIDENCE_THRESHOLD,
            )
            self._yolo_detector = YOLODetector.load(
                model_path,
                device=device,
                conf_threshold=YOLO_CONFIDENCE_THRESHOLD,
            )
            self._set_detection_state(DetectionHealthStatus.ACTIVE)
            logger.info(
                "detection_worker: loaded YOLODetector model_path=%s device=%s",
                model_path,
                device,
            )
        except Exception:
            self._yolo_detector = None
            self._set_detection_state(
                DetectionHealthStatus.UNAVAILABLE,
                "Detection model unavailable",
            )
            logger.exception(
                "detection_worker: failed to load YOLODetector model_path=%s device=%s",
                model_path,
                device,
            )

    def _load_rfdetr_detector(self) -> None:
        """Load RF-DETR detector if configured."""
        if self._rfdetr_detector is not None:
            return

        model_path = self._configured_rfdetr_model_path()
        if not model_path:
            logger.debug("detection_worker: RFDETR_MODEL_PATH not configured; RF-DETR disabled")
            return

        device = self._configured_device()
        try:
            from core.infer.rfdetr_detector import RFDETRDetector

            logger.info(
                "detection_worker: loading RFDETRDetector model_path=%s device=%s conf_threshold=%.3f",
                model_path,
                device,
                YOLO_CONFIDENCE_THRESHOLD,
            )
            self._rfdetr_detector = RFDETRDetector.load(
                model_path,
                device=device,
                conf_threshold=YOLO_CONFIDENCE_THRESHOLD,
            )
            logger.info(
                "detection_worker: loaded RFDETRDetector model_path=%s device=%s",
                model_path,
                device,
            )
            if self._yolo_detector is None:
                self._set_detection_state(DetectionHealthStatus.ACTIVE)
        except Exception:
            self._rfdetr_detector = None
            logger.exception(
                "detection_worker: failed to load RFDETRDetector model_path=%s device=%s",
                model_path,
                device,
            )

    async def _run(self) -> None:
        logger.info("detection_worker: started for source=%s", SOURCE_ID)
        loop = asyncio.get_running_loop()

        # Load models in thread pool so the event loop stays responsive during init
        await loop.run_in_executor(None, self._load_detector)
        await loop.run_in_executor(None, self._load_rfdetr_detector)

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
                    self._sync_detection_state_for_frame_source(has_frame=False)
                    self._refresh_detection_state(last_frame_time)
                    await self._broadcast_status()
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0, detection_interval - elapsed))
                    continue

                self._sync_detection_state_for_frame_source(has_frame=True)

                self._frame_counter += 1
                video_frame = VideoFrame(
                    frame_id=raw_frame.sequence_number,
                    captured_at=raw_frame.captured_at,
                    source_width=raw_frame.source_width,
                    source_height=raw_frame.source_height,
                    image=raw_frame.image,
                )

                primary_detector = self._yolo_detector or self._rfdetr_detector
                primary_model_name = "yolo" if self._yolo_detector is not None else "rfdetr"

                if primary_detector is None:
                    await self._broadcast_status()
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0, detection_interval - elapsed))
                    continue

                if not self._should_process_frame(video_frame.frame_id):
                    self._refresh_detection_state(last_frame_time)
                    await self._broadcast_status()
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0, detection_interval - elapsed))
                    continue

                # Run primary detector inference in thread pool.
                try:
                    detect_fn = functools.partial(
                        primary_detector.detect,
                        video_frame.image,
                        frame_id=video_frame.frame_id,
                    )
                    primary_result: DetectionResult = await loop.run_in_executor(None, detect_fn)
                except Exception:
                    if primary_model_name == "yolo":
                        self._yolo_detector = None
                    else:
                        self._rfdetr_detector = None
                    self._set_detection_state(
                        DetectionHealthStatus.UNAVAILABLE,
                        "Detection model unavailable",
                    )
                    logger.exception("detection_worker: %s inference failed", primary_model_name)
                    await self._broadcast_status()
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0, detection_interval - elapsed))
                    continue

                # Override captured_at with frame's acquisition time
                primary_result.captured_at = video_frame.captured_at
                primary_result.detections = self._validated_detections(primary_result.detections)

                if self._session and self._session.detection_status == DetectionHealthStatus.UNAVAILABLE:
                    await self._broadcast_status()
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0, detection_interval - elapsed))
                    continue

                # Run RF-DETR inference if available
                rfdetr_result: Optional[DetectionResult] = None
                if self._yolo_detector is not None and self._rfdetr_detector is not None:
                    try:
                        detect_fn = functools.partial(
                            self._rfdetr_detector.detect,
                            video_frame.image,
                            frame_id=video_frame.frame_id,
                        )
                        rfdetr_result = await loop.run_in_executor(None, detect_fn)
                        rfdetr_result.captured_at = video_frame.captured_at
                    except Exception:
                        logger.exception("detection_worker: RF-DETR inference failed")
                        self._rfdetr_detector = None

                feed_snapshot = self._feed_analyzer.analyze_frame(primary_result, model_name=primary_model_name)

                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                if self._session:
                    self._session.last_overlay_at = datetime.now(timezone.utc)
                    self._record_overlay_timing(time.monotonic())
                    self._session.detection_status = derive_detection_status(
                        last_overlay_at=self._session.last_overlay_at,
                        last_frame_at=last_frame_time,
                        stale_seconds=STALE_OVERLAY_MS / 1000,
                    )

                await self._broadcast_overlay(
                    video_frame,
                    primary_result,
                    rfdetr_result,
                    feed_snapshot,
                    now_ms,
                    primary_model_name=primary_model_name,
                )

                elapsed = time.monotonic() - loop_start
                await asyncio.sleep(max(0, detection_interval - elapsed))

        except asyncio.CancelledError:
            logger.info("detection_worker: cancelled")
        except Exception:
            logger.exception("detection_worker: unhandled error")
        finally:
            await self._shutdown()

    async def _broadcast_overlay(
        self,
        frame: VideoFrame,
        primary_result: DetectionResult,
        rfdetr_result: Optional[DetectionResult],
        feed_snapshot: "FeedAreaSnapshot",
        now_ms: int,
        primary_model_name: str = "yolo",
    ) -> None:
        if self._session is None:
            return

        primary_payloads = [
            DetectionPayload(
                class_name=d.class_name,
                confidence=d.confidence,
                x=d.bbox_x,
                y=d.bbox_y,
                width=d.bbox_w,
                height=d.bbox_h,
            )
            for d in primary_result.detections
        ]

        # Build detector results list
        detectors = [
            DetectorResult(
                model_name=primary_model_name,
                inference_ms=primary_result.inference_ms,
                detections=primary_payloads,
            )
        ]

        # Add RF-DETR if available
        if rfdetr_result:
            rfdetr_payloads = [
                DetectionPayload(
                    class_name=d.class_name,
                    confidence=d.confidence,
                    x=d.bbox_x,
                    y=d.bbox_y,
                    width=d.bbox_w,
                    height=d.bbox_h,
                )
                for d in rfdetr_result.detections
            ]
            detectors.append(
                DetectorResult(
                    model_name="rfdetr",
                    inference_ms=rfdetr_result.inference_ms,
                    detections=rfdetr_payloads,
                )
            )

        # Feed area data
        feed_area = FeedAreaData(
            raw_area=feed_snapshot.raw_area,
            kalman_area=feed_snapshot.kalman_area,
            detection_count=feed_snapshot.detection_count,
        )

        # Feed analysis history (last 10 seconds at 30fps)
        frame_ids, raw_areas, kalman_areas = self._feed_analyzer.get_history_data()
        feed_analysis = FeedAnalysisHistory(
            frame_ids=frame_ids,
            raw_areas=raw_areas,
            kalman_areas=kalman_areas,
        )

        msg = DetectionsMessage(
            schema="detections.v2",
            source_id=self._session.source_id,
            frame_id=frame.frame_id,
            timestamp_ms=now_ms,
            source_width=frame.source_width,
            source_height=frame.source_height,
            stream_status=self._session.stream_status,
            detection_status=self._session.detection_status,
            detections=primary_payloads,
            detectors=detectors,
            feed_area=feed_area,
            feed_analysis=feed_analysis,
        )
        overlay_hub.broadcast(msg.to_ws_dict())

    def _validated_detections(self, detections: list[Detection]) -> list[Detection]:
        supported = set(YOLO_SUPPORTED_CLASSES)
        for detection in detections:
            if detection.class_name not in supported:
                self._set_detection_state(
                    DetectionHealthStatus.UNAVAILABLE,
                    "Detection model class mismatch",
                )
                logger.error("detection_worker: unsupported YOLO class %s", detection.class_name)
                return []
        return list(detections)

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
