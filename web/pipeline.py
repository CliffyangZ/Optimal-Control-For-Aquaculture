"""Inference pipeline: decouples RTSP frame capture from RF-DETR inference.

Two background threads run independently:
  • LatestFrameReader  — pulls RTSP frames as fast as the source allows
  • _infer_loop        — runs RF-DETR at maximum throughput, capped at 30 FPS

The Streamlit display loop reads `get_latest()` at its own cadence without
ever blocking on inference or stream acquisition.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.types import DetectionResult

# Per-class BGR colours for bbox overlay
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "fish":     (0, 220, 80),   # green
    "feed":     (0, 160, 255),  # orange
    "lumpfish": (255, 80, 0),   # blue
    "debris":   (128, 0, 180),  # purple
}
_DEFAULT_COLOR: tuple[int, int, int] = (0, 220, 220)  # cyan

# Target inference cadence (max 30 FPS to avoid spinning)
_INFER_INTERVAL = 1.0 / 30.0


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    frame: np.ndarray                  # BGR image from RTSP
    result: Optional[DetectionResult]  # None until first inference completes
    display_fps: float = 0.0           # frames rendered per second by UI
    infer_fps: float = 0.0             # inference iterations per second
    inference_ms: float = 0.0
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Manages stream reader + detector; thread-safe result handoff to UI."""

    def __init__(
        self,
        stream_url: str,
        model_path: str | Path,
        device: str,
        conf_threshold: float,
        *,
        use_trt: bool = True,
    ) -> None:
        self._stream_url = stream_url
        self._model_path = Path(model_path)
        self._device = device
        self._conf_threshold = conf_threshold
        self._use_trt = use_trt

        self._reader = None
        self._detector = None
        self._opt_info = None

        self._latest: Optional[FrameResult] = None
        self._lock = threading.Lock()
        self._running = False
        self._infer_thread: Optional[threading.Thread] = None

        # FPS tracking
        self._infer_count = 0
        self._infer_fps = 0.0
        self._fps_t0 = 0.0

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        from streaming.latest_frame_reader import LatestFrameReader
        from web.trt import load_optimized_detector

        self._detector, self._opt_info = load_optimized_detector(
            self._model_path,
            device=self._device,
            conf_threshold=self._conf_threshold,
            use_trt=self._use_trt,
        )

        self._reader = LatestFrameReader(self._stream_url)
        self._reader.start()

        self._running = True
        self._fps_t0 = time.monotonic()
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True, name="rfdetr-infer")
        self._infer_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._reader:
            self._reader.stop()
        if self._infer_thread:
            self._infer_thread.join(timeout=3)
        self._latest = None

    @property
    def opt_info(self):
        return self._opt_info

    def get_latest(self) -> Optional[FrameResult]:
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    # Background inference loop
    # ------------------------------------------------------------------

    def _infer_loop(self) -> None:
        last_result: Optional[DetectionResult] = None

        while self._running:
            t_loop_start = time.monotonic()

            frame_data = self._reader.get_latest_frame() if self._reader else None
            if frame_data is None:
                time.sleep(0.02)
                continue

            try:
                result = self._detector.detect(frame_data.image, frame_id=frame_data.sequence_number)
                last_result = result
            except Exception:
                result = last_result  # keep last good result on transient error

            self._infer_count += 1
            now = time.monotonic()
            elapsed = now - self._fps_t0
            if elapsed >= 1.0:
                self._infer_fps = self._infer_count / elapsed
                self._infer_count = 0
                self._fps_t0 = now

            with self._lock:
                self._latest = FrameResult(
                    frame=frame_data.image.copy(),
                    result=result,
                    infer_fps=self._infer_fps,
                    inference_ms=result.inference_ms if result else 0.0,
                    captured_at=frame_data.captured_at,
                )

            # Pace to at most 30 FPS
            spent = time.monotonic() - t_loop_start
            gap = _INFER_INTERVAL - spent
            if gap > 0:
                time.sleep(gap)


# ---------------------------------------------------------------------------
# BBox overlay rendering
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    result: Optional[DetectionResult],
    *,
    show_labels: bool = True,
    show_conf: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Return a copy of *frame* with bounding boxes drawn from *result*."""
    out = frame.copy()
    if result is None or not result.detections:
        return out

    h, w = out.shape[:2]

    for det in result.detections:
        color = _CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)

        x1 = int(det.bbox_x * w)
        y1 = int(det.bbox_y * h)
        x2 = int((det.bbox_x + det.bbox_w) * w)
        y2 = int((det.bbox_y + det.bbox_h) * h)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        if show_labels:
            label = det.class_name
            if show_conf:
                label = f"{label} {det.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            ty = max(y1 - 4, lh + 4)
            cv2.rectangle(out, (x1, ty - lh - baseline - 2), (x1 + lw + 2, ty), color, -1)
            cv2.putText(
                out, label,
                (x1 + 1, ty - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

    return out
