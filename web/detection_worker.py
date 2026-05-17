"""Shared detection worker lifecycle for one configured stream.

Bridges the new spatiotemporal ripple detector (`core.ripple_algo.RippleDetector`)
with the FastAPI / WebSocket overlay layer. Each frame produced by
`LatestFrameReader` is fed to `RippleDetector.process()`, and the resulting
`ActivityResult` is converted into an `OverlayMessage` containing:

  * The headline `activity` badge (level, integer score, confidence, raw and
    smoothed energy) so the frontend energy chart can plot it in real time.
  * A list of normalized candidate `regions` derived from the detector's
    candidate ripple centres, for the SVG overlay layer.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from config.setting import (
    IDLE_SHUTDOWN_DELAY_SECONDS,
    SOURCE_ID,
    STREAM_URL,
    TARGET_DETECTION_FPS,
    VIEWER_HEARTBEAT_TIMEOUT_SECONDS,
)
from core.models import (
    ActivityBadge,
    DetectionHealthStatus,
    LiveStreamSession,
    StreamStatus,
    VideoFrame,
    ViewerSession,
    ViewerState,
)
from core.ripple_algo import ActivityResult, RippleAlgoConfig, RippleDetector, activity_level
from core.stream_health import derive_detection_status, derive_stream_status
from streaming.latest_frame_reader import LatestFrameReader
from web.overlay_hub import overlay_hub
from web.schemas import (
    ActivityPayload,
    OverlayMessage,
    RegionPayload,
    StatusMessage,
)

logger = logging.getLogger(__name__)


# Map the new 5-level vocabulary from `core.ripple_algo.activity_level` to
# the 3-level vocabulary expected by the frontend's existing ActivityBadge.
_LEVEL_TO_BADGE: dict[str, str] = {
    "idle": "Low",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "intense": "High",
}


def _badge_level(energy_label: str) -> str:
    return _LEVEL_TO_BADGE.get(energy_label, "Low")


def _candidate_radius_px(cfg: RippleAlgoConfig) -> int:
    """Visual bounding-box half-size around each ripple candidate centre."""
    return max(8, int(round(cfg.candidate_sigma * 1.5)))


def _candidates_to_regions(
    result: ActivityResult,
    cfg: RippleAlgoConfig,
    source_width: int,
    source_height: int,
) -> list[RegionPayload]:
    """Convert ripple candidate centres into normalized overlay regions.

    Candidates (cx, cy) are in the detector's internal pixel space, which may
    differ from source_width × source_height when cfg.input_scale != 1.0.
    We normalise against the actual processed dimensions (result.heatmap.shape)
    so the [0, 1] coordinates are always correct regardless of input_scale.
    source_width / source_height are kept in the signature for compatibility but
    are no longer used for normalisation.
    """
    if not result.candidates:
        return []

    half = _candidate_radius_px(cfg)
    # Use the actual processed-frame dimensions, not the pre-scale source dims.
    det_h, det_w = result.heatmap.shape[:2]
    sw = max(1, det_w)
    sh = max(1, det_h)
    regions: list[RegionPayload] = []
    for i, (cx, cy, score) in enumerate(result.candidates):
        if score <= 0.0:
            continue
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(sw, cx + half)
        y1 = min(sh, cy + half)
        if x1 <= x0 or y1 <= y0:
            continue
        regions.append(
            RegionPayload(
                id=f"candidate-{i + 1}",
                x=x0 / sw,
                y=y0 / sh,
                width=(x1 - x0) / sw,
                height=(y1 - y0) / sh,
                score=float(min(1.0, max(0.0, score))),
            )
        )
    return regions


def _activity_payload(
    result: ActivityResult,
    cfg: RippleAlgoConfig,
) -> ActivityPayload:
    """Build the headline activity payload from the latest ActivityResult.

    `score` is the smoothed energy mapped to the legacy 0–100 integer scale so
    existing frontend widgets keep working; `energy` carries the original
    0.0–1.0 value for the time-series chart.
    """
    energy = float(max(0.0, min(1.0, result.energy)))
    energy_raw = float(max(0.0, min(1.0, result.energy_raw)))
    score = int(round(energy * 100))
    # Confidence proxy: how strongly the detector "committed" this frame.
    # When energy_raw is much lower than energy_ema we are still in the
    # release tail (decaying) — slightly lower confidence; when raw > ema we
    # are on a rising edge — high confidence.
    if energy <= 0.0:
        confidence = 0.0
    else:
        ratio = energy_raw / max(energy, 1e-6)
        confidence = float(max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(ratio - 0.5))))
    return ActivityPayload(
        level=_badge_level(activity_level(energy)),
        score=score,
        confidence=confidence,
        energy=energy,
        energy_raw=energy_raw,
        frame_index=int(result.frame_index),
    )


class DetectionWorker:
    """Single shared detection loop for the configured stream.

    Lifecycle:
      * Starts when the first viewer joins via `register_viewer()`.
      * Runs at TARGET_DETECTION_FPS, broadcasting overlays through `overlay_hub`.
      * Idles down when `active_viewer_count` stays at 0 for
        `IDLE_SHUTDOWN_DELAY_SECONDS`.
    """

    def __init__(self) -> None:
        self._session: Optional[LiveStreamSession] = None
        self._viewers: dict[str, ViewerSession] = {}
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._frame_reader: Optional[LatestFrameReader] = None
        # New spatiotemporal detector. We keep a single configured instance for
        # the worker's lifetime so its sliding windows and tracker state remain
        # warm across frames.
        self._cfg = RippleAlgoConfig()
        self._detector = RippleDetector(self._cfg)
        self._frame_counter = 0
        self._detection_times: list[float] = []
        self._last_processed_seq: int = -1
        # Wall-clock time of the most recently processed frame. A large gap
        # (> _STREAM_GAP_RESET_S) indicates the RTSP source restarted and the
        # detector's inter-frame state (prev_gray, tracked centres, EMA) would
        # compare the new first frame against the old last frame, producing a
        # spurious energy spike. Resetting on gap avoids this artefact.
        self._last_processed_frame_at: Optional[datetime] = None
        self._STREAM_GAP_RESET_S: float = 3.0

    # --- Public API ---

    @property
    def session(self) -> Optional[LiveStreamSession]:
        return self._session

    async def register_viewer(
        self, viewer_id: str, overlay_enabled: bool = True
    ) -> LiveStreamSession:
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
                    detection_status=DetectionHealthStatus.UNAVAILABLE,
                )
                # Reset detector state so per-session sliding windows start
                # from a clean slate (otherwise mid/long scores carry state
                # from the previous session).
                self._detector.reset()
                self._frame_counter = 0
                self._detection_times.clear()
                self._last_processed_seq = -1
                self._last_processed_frame_at = None
                self._start_worker()
            self._session.active_viewer_count = len(self._active_viewers())
            return self._session

    async def unregister_viewer(
        self, viewer_id: str
    ) -> Optional[LiveStreamSession]:
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

    # --- Internal helpers ---

    def _active_viewers(self) -> list[ViewerSession]:
        return [
            v
            for v in self._viewers.values()
            if v.state
            in (ViewerState.ACTIVE, ViewerState.CONNECTING, ViewerState.RECONNECTING)
        ]

    def _start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._frame_reader = LatestFrameReader(STREAM_URL)
            self._frame_reader.start()
            self._worker_task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        logger.info("detection_worker: started for source=%s", SOURCE_ID)
        detection_interval = (
            1.0 / TARGET_DETECTION_FPS if TARGET_DETECTION_FPS > 0 else 0.1
        )
        idle_since: Optional[float] = None

        try:
            while True:
                loop_start = time.monotonic()

                # Idle-shutdown when there are no active viewers for too long.
                active = self._active_viewers()
                if not active:
                    if idle_since is None:
                        idle_since = time.monotonic()
                    elif time.monotonic() - idle_since >= IDLE_SHUTDOWN_DELAY_SECONDS:
                        logger.info(
                            "detection_worker: no active viewers; shutting down"
                        )
                        break
                else:
                    idle_since = None

                # Heartbeat timeout cleanup.
                now = datetime.now(timezone.utc)
                async with self._lock:
                    for viewer_id, v in list(self._viewers.items()):
                        age = (now - v.last_heartbeat_at).total_seconds()
                        if age > VIEWER_HEARTBEAT_TIMEOUT_SECONDS:
                            logger.info(
                                "detection_worker: heartbeat timeout viewer=%s",
                                viewer_id,
                            )
                            v.state = ViewerState.TIMED_OUT
                            self._viewers.pop(viewer_id, None)
                    if self._session:
                        self._session.active_viewer_count = len(
                            self._active_viewers()
                        )

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
                    await asyncio.sleep(max(0.0, detection_interval - elapsed))
                    continue

                # Skip duplicate frames: when RTSP delivers slower than the
                # detection rate, get_latest_frame() returns the same frame on
                # consecutive ticks. Processing the same image twice makes the
                # detector compute diff=0 on tick N+1, gate fires → energy=0,
                # then diff≠0 on tick N+2 → energy spikes back. This produces
                # the "jittery idle" artefact on static / slow-stream scenes.
                if raw_frame.sequence_number == self._last_processed_seq:
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0.0, detection_interval - elapsed))
                    continue
                self._last_processed_seq = raw_frame.sequence_number

                self._frame_counter += 1
                video_frame = VideoFrame(
                    frame_id=raw_frame.sequence_number,
                    captured_at=raw_frame.captured_at,
                    source_width=raw_frame.source_width,
                    source_height=raw_frame.source_height,
                    image=raw_frame.image,
                )

                # Reset detector when the RTSP source restarts. A large gap
                # between successive processed frames means LatestFrameReader
                # reconnected: the detector would otherwise compare the new
                # first frame against the stale last frame and generate a
                # spurious energy spike (scene-change artefact).
                if (
                    self._last_processed_frame_at is not None
                    and video_frame.captured_at is not None
                ):
                    gap = (
                        video_frame.captured_at - self._last_processed_frame_at
                    ).total_seconds()
                    if gap > self._STREAM_GAP_RESET_S:
                        logger.info(
                            "detection_worker: %.1fs frame gap — resetting detector",
                            gap,
                        )
                        self._detector.reset()
                self._last_processed_frame_at = video_frame.captured_at

                # Run detection off the event loop — OpenCV work is heavy and
                # synchronous.
                loop = asyncio.get_event_loop()
                t0 = time.monotonic()
                result = await loop.run_in_executor(
                    None, self._detector.process, video_frame.image
                )
                detection_time = time.monotonic() - t0
                self._detection_times.append(detection_time)
                if len(self._detection_times) > 30:
                    self._detection_times.pop(0)

                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                if self._session:
                    self._session.last_overlay_at = datetime.now(timezone.utc)
                    avg_dt = (
                        sum(self._detection_times) / len(self._detection_times)
                        if self._detection_times
                        else 0.0
                    )
                    self._session.detection_fps = (
                        round(1.0 / avg_dt, 1) if avg_dt > 0 else 0.0
                    )
                    self._session.detection_status = derive_detection_status(
                        last_overlay_at=self._session.last_overlay_at,
                        last_frame_at=last_frame_time,
                    )

                await self._broadcast_overlay(video_frame, result, now_ms)

                elapsed = time.monotonic() - loop_start
                await asyncio.sleep(max(0.0, detection_interval - elapsed))

        except asyncio.CancelledError:
            logger.info("detection_worker: cancelled")
        except Exception:
            logger.exception("detection_worker: unhandled error")
        finally:
            await self._shutdown()

    async def _broadcast_overlay(
        self,
        frame: VideoFrame,
        result: Optional[ActivityResult],
        now_ms: int,
    ) -> None:
        if self._session is None:
            return

        regions: list[RegionPayload] = []
        activity: Optional[ActivityPayload] = None
        message: Optional[str] = None
        det_status = self._session.detection_status

        if result is None:
            # The detector returns None on the very first frame (it needs a
            # previous frame to compute temporal features). Treat that as a
            # transient "warming up" state.
            det_status = DetectionHealthStatus.UNAVAILABLE
            message = "Detection warming up"
        else:
            regions = _candidates_to_regions(
                result, self._cfg, frame.source_width, frame.source_height
            )
            activity = _activity_payload(result, self._cfg)

        msg = OverlayMessage(
            schema="overlay.v1",
            source_id=self._session.source_id,
            frame_id=frame.frame_id,
            timestamp_ms=now_ms,
            source_width=frame.source_width,
            source_height=frame.source_height,
            stream_status=self._session.stream_status,
            detection_status=det_status,
            activity=activity,
            regions=regions,
            message=message,
        )
        overlay_hub.broadcast(msg.to_ws_dict())

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
