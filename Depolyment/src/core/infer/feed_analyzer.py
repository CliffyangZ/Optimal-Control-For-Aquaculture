"""Feed area analysis and calculation from detection results."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.types import Detection, DetectionResult
from core.infer.kalman_filter import create_kalman_filter, KalmanFilterConfig

_RATE_WINDOW = 30
_HUNGER_MILD_THRESHOLD = -0.05
_HUNGER_SEVERE_THRESHOLD = -0.10


def _assess_hunger(window_rate_avg: float) -> str:
    if window_rate_avg > _HUNGER_MILD_THRESHOLD:
        return "Mild"
    elif window_rate_avg > _HUNGER_SEVERE_THRESHOLD:
        return "Moderate"
    else:
        return "Severe"


@dataclass
class FeedAreaSnapshot:
    """Single frame's feed area analysis."""

    frame_id: int
    timestamp_ms: int
    raw_area: float
    kalman_area: float
    ema_area: float
    consumption_rate: float
    """Window-averaged log-space consumption rate (negative = feed being eaten)."""
    hunger_level: str
    """'Mild' | 'Moderate' | 'Severe'"""
    detection_count: int


@dataclass
class FeedAnalysis:
    """Real-time feed area analysis with Kalman filtering, online EMA, and hunger assessment."""

    config: KalmanFilterConfig = field(default_factory=KalmanFilterConfig)
    history: deque = field(default_factory=lambda: deque(maxlen=300))  # 10s at 30fps

    def __post_init__(self) -> None:
        self._kf = create_kalman_filter(self.config)
        self._ema_log: float = 0.0
        self._ema_initialized: bool = False
        self._prev_ema_log: float = 0.0
        self._rate_window: deque = deque(maxlen=_RATE_WINDOW)

    def analyze_frame(self, result: DetectionResult) -> FeedAreaSnapshot:
        """Analyze a single detection result and return feed area snapshot."""
        feed_detections = [d for d in result.detections if d.class_name == "feed"]
        raw_area = self._calculate_total_area(
            feed_detections, result.source_width, result.source_height
        )

        kalman_area = self._kf.update(
            raw_area,
            reject_outliers=self.config.reject_outliers,
            outlier_threshold=self.config.outlier_threshold,
        )

        # Online EMA in log-space applied to Kalman estimate
        alpha = self._kf.ema_alpha
        if kalman_area > 0:
            log_val = math.log(kalman_area)
        elif self._ema_initialized:
            log_val = self._ema_log  # hold last value when feed absent
        else:
            log_val = 0.0

        if not self._ema_initialized and kalman_area > 0:
            self._ema_log = log_val
            self._ema_initialized = True
        elif self._ema_initialized:
            self._ema_log = alpha * log_val + (1.0 - alpha) * self._ema_log

        ema_area = math.exp(self._ema_log) if self._ema_initialized else 0.0

        # Consumption rate: change in log-EMA (negative → feed being consumed)
        inst_rate = self._ema_log - self._prev_ema_log
        self._prev_ema_log = self._ema_log
        self._rate_window.append(inst_rate)

        window_rate = (
            sum(self._rate_window) / len(self._rate_window) if self._rate_window else 0.0
        )
        hunger = _assess_hunger(window_rate)

        snapshot = FeedAreaSnapshot(
            frame_id=result.frame_id,
            timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            raw_area=raw_area,
            kalman_area=kalman_area,
            ema_area=ema_area,
            consumption_rate=window_rate,
            hunger_level=hunger,
            detection_count=len(feed_detections),
        )
        self.history.append(snapshot)
        return snapshot

    def get_chart_data(self) -> dict:
        """Return arrays ready for real-time chart rendering."""
        if not self.history:
            return {
                "frame_ids": [],
                "raw": [],
                "kalman": [],
                "ema": [],
                "consumption_rate": [],
                "hunger": [],
            }
        return {
            "frame_ids": [s.frame_id for s in self.history],
            "raw": [s.raw_area for s in self.history],
            "kalman": [s.kalman_area for s in self.history],
            "ema": [s.ema_area for s in self.history],
            "consumption_rate": [s.consumption_rate for s in self.history],
            "hunger": [s.hunger_level for s in self.history],
        }

    def get_latest_snapshot(self) -> FeedAreaSnapshot | None:
        """Get the most recent analysis snapshot."""
        return self.history[-1] if self.history else None

    def _calculate_total_area(
        self, detections: list[Detection], width: int, height: int
    ) -> float:
        total = 0.0
        for d in detections:
            total += (d.bbox_w * width) * (d.bbox_h * height)
        return total

    def reset(self) -> None:
        """Reset all filter and history state."""
        self._kf = create_kalman_filter(self.config)
        self.history.clear()
        self._ema_log = 0.0
        self._ema_initialized = False
        self._prev_ema_log = 0.0
        self._rate_window.clear()
