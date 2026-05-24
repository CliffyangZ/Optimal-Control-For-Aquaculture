"""Feed area analysis and calculation from detection results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque

from core.types import Detection, DetectionResult
from core.infer.kalman_filter import create_kalman_filter, KalmanFilterConfig


@dataclass
class FeedAreaSnapshot:
    """Single frame's feed area analysis."""

    frame_id: int
    timestamp_ms: int
    raw_area: float
    """Calculated area from feed detections."""

    kalman_area: float
    """Kalman-filtered area estimate."""

    detection_count: int
    """Number of feed detections in this frame."""


@dataclass
class FeedAnalysis:
    """Real-time feed area analysis with Kalman filtering."""

    config: KalmanFilterConfig = field(default_factory=KalmanFilterConfig)
    kalman_filter: KalmanFilterConfig = field(init=False)
    history: deque = field(default_factory=lambda: deque(maxlen=300))  # 10s at 30fps

    def __post_init__(self) -> None:
        self.kalman_filter = create_kalman_filter(self.config)

    def analyze_frame(self, result: DetectionResult, model_name: str = "yolo") -> FeedAreaSnapshot:
        """Analyze a single detection result and return feed area snapshot."""
        # Calculate raw feed area from detections
        feed_detections = [d for d in result.detections if d.class_name == "feed"]
        raw_area = self._calculate_total_area(feed_detections, result.source_width, result.source_height)

        # Update Kalman filter with raw measurement
        kalman_area = self.kalman_filter.update(
            raw_area,
            reject_outliers=self.config.reject_outliers,
            outlier_threshold=self.config.outlier_threshold,
        )

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        snapshot = FeedAreaSnapshot(
            frame_id=result.frame_id,
            timestamp_ms=now_ms,
            raw_area=raw_area,
            kalman_area=kalman_area,
            detection_count=len(feed_detections),
        )

        self.history.append(snapshot)
        return snapshot

    def _calculate_total_area(self, detections: list[Detection], width: int, height: int) -> float:
        """Calculate total area covered by feed detections in pixels."""
        total_area = 0.0
        for d in detections:
            # Convert normalized coordinates to pixels
            pixel_w = d.bbox_w * width
            pixel_h = d.bbox_h * height
            area = pixel_w * pixel_h
            total_area += area
        return total_area

    def get_history_data(self) -> tuple[list[int], list[float], list[float]]:
        """Get frame IDs, raw areas, and Kalman-filtered areas for plotting."""
        if not self.history:
            return [], [], []

        frame_ids = [s.frame_id for s in self.history]
        raw_areas = [s.raw_area for s in self.history]
        kalman_areas = [s.kalman_area for s in self.history]

        return frame_ids, raw_areas, kalman_areas

    def get_latest_snapshot(self) -> FeedAreaSnapshot | None:
        """Get the most recent analysis snapshot."""
        return self.history[-1] if self.history else None

    def reset(self) -> None:
        """Reset the analyzer state."""
        self.kalman_filter = create_kalman_filter(self.config)
        self.history.clear()
