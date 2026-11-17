"""Kalman Filter for smoothing feed area estimates with outlier rejection and RTS smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class KalmanFilterConfig:
    """Configuration for Kalman Filter behavior."""

    sigma_R: float = 0.5
    """Measurement noise std dev in log-space (observational error). Increased to allow natural area variations."""

    sigma_Q: float = 0.3
    """Process noise std dev in log-space (model uncertainty)."""

    reject_outliers: bool = True
    """Whether to reject measurements >N*sigma from predicted value."""

    outlier_threshold: float = 2.5
    """Standard deviations threshold for outlier rejection."""

    use_log_space: bool = True
    """Whether to work in log-space (log-normal filter)."""

    max_missing_frames: int = 5
    """Number of consecutive zero-detection frames before allowing P to grow.
    Treats brief detector failures as glitches: estimate held fixed, uncertainty frozen."""

    ema_alpha: float = 0.1
    """EMA smoothing factor applied after RTS in log-space. Range (0, 1).
    Lower = heavier smoothing (slower response). 0.1 ≈ ~19-frame half-life at 30fps."""


class KalmanFilter:
    """1D Kalman Filter for continuous state estimation.

    Filters noisy measurements of feed area using a linear Gaussian model.
    Supports zero-value skipping and outlier rejection.
    """

    def __init__(self, sigma_R: float, sigma_Q: float) -> None:
        """Initialize filter state.

        Args:
            sigma_R: Measurement noise std dev
            sigma_Q: Process noise std dev
        """
        self.sigma_R = sigma_R
        self.sigma_Q = sigma_Q

        self.R = sigma_R**2
        self.Q = sigma_Q**2

        self.x_estimate = 0.0
        self.p_estimate = 1.0

        self.initialized = False
        self.history_x: list[list[float]] = []
        self.history_p: list[list[float]] = []
        self.history_measurements: list[float] = []

    def update(self, measurement: float, reject_outliers: bool = False, outlier_threshold: float = 3.0) -> float:
        """Update filter with a new measurement.

        Args:
            measurement: Raw measurement value
            reject_outliers: Whether to skip outlier measurements
            outlier_threshold: Reject measurements >threshold*sigma from prediction

        Returns:
            Filtered estimate after update
        """
        self.history_measurements.append(measurement)

        if measurement <= 0:
            self.history_x.append([self.x_estimate])
            self.history_p.append([self.p_estimate])
            return self.x_estimate if self.initialized else 0.0

        if not self.initialized:
            self.x_estimate = measurement
            self.p_estimate = self.R
            self.initialized = True
            self.history_x.append([self.x_estimate])
            self.history_p.append([self.p_estimate])
            return self.x_estimate

        y = measurement
        predicted_y = self.x_estimate
        S = self.p_estimate + self.R

        if reject_outliers:
            innovation = abs(y - predicted_y)
            if innovation > outlier_threshold * math.sqrt(S):
                self.history_x.append([self.x_estimate])
                self.history_p.append([self.p_estimate])
                return self.x_estimate

        K = self.p_estimate / S
        self.x_estimate = self.x_estimate + K * (y - self.x_estimate)
        self.p_estimate = (1 - K) * self.p_estimate

        self.history_x.append([self.x_estimate])
        self.history_p.append([self.p_estimate])

        return self.x_estimate

    def predict(self) -> float:
        """Predict next state without measurement update."""
        self.p_estimate = self.p_estimate + self.Q
        self.history_x.append([self.x_estimate])
        self.history_p.append([self.p_estimate])
        return self.x_estimate

    def rts_smooth(self) -> list[float]:
        """Rauch-Tung-Striebel backward smoothing.

        Post-processes entire trajectory using forward-backward information
        to reduce bias and spikes. Must be called after all forward updates.

        Returns:
            Smoothed state estimates
        """
        n = len(self.history_x)
        if n == 0:
            return []

        x_smooth = [[self.history_x[i][0]] for i in range(n)]
        p_smooth = [self.history_p[i][0] for i in range(n)]

        for k in range(n - 2, -1, -1):
            x_pred = self.history_x[k][0]
            p_pred = self.history_p[k][0] + self.Q
            C = self.history_p[k][0] / p_pred

            x_smooth[k][0] = self.history_x[k][0] + C * (x_smooth[k + 1][0] - x_pred)
            p_smooth[k] = self.history_p[k][0] + C**2 * (p_smooth[k + 1] - p_pred)

        self.history_x = x_smooth
        self.history_p = [[p] for p in p_smooth]

        return [s[0] for s in x_smooth]


class LogNormalKalmanFilter(KalmanFilter):
    """Kalman Filter for log-normal distributed measurements.

    Feed area measurements follow a log-normal distribution.
    This filter operates in log-space for better numerical stability
    but stores values in original space for consistency.
    """

    def __init__(self, sigma_R: float, sigma_Q: float) -> None:
        """Initialize filter state.

        Args:
            sigma_R: Measurement noise std dev (in log-space)
            sigma_Q: Process noise std dev (in log-space)
        """
        super().__init__(sigma_R, sigma_Q)
        self.history_x_log: list[float] = []
        self.history_p_pred: list[float] = []  # predicted P before measurement update
        self._init_frame: int = -1              # index of first valid measurement
        self._consecutive_missing: int = 0      # consecutive frames with z_t = 0
        self.max_missing_frames: int = 5        # set via create_kalman_filter
        self.ema_alpha: float = 0.1             # set via create_kalman_filter
        self.history_ema_log: list[float] = []  # EMA values in log-space (populated by apply_ema)

    def update(self, area_measurement: float, reject_outliers: bool = False, outlier_threshold: float = 3.0) -> float:
        """Update filter with log-transformed measurement.

        Args:
            area_measurement: Raw area measurement
            reject_outliers: Whether to skip outlier measurements
            outlier_threshold: Reject measurements >threshold*sigma from prediction

        Returns:
            Filtered area estimate in original space
        """
        if area_measurement <= 0:
            self._consecutive_missing += 1
            if self.initialized:
                # Freeze uncertainty during brief gaps: treat as detector glitch, not real absence.
                # Only let P grow after max_missing_frames consecutive zeros (feed genuinely gone).
                if self._consecutive_missing <= self.max_missing_frames:
                    p_pred = self.p_estimate  # frozen: no uncertainty growth
                else:
                    p_pred = self.p_estimate + self.Q
                self.p_estimate = p_pred
            else:
                p_pred = self.p_estimate
            self.history_measurements.append(0.0)
            filtered_area = math.exp(self.x_estimate) if self.initialized else 0.0
            self.history_x.append([filtered_area])
            self.history_x_log.append(self.x_estimate)
            self.history_p.append([self.p_estimate])
            self.history_p_pred.append(p_pred)
            return filtered_area

        log_measurement = math.log(area_measurement)
        self._consecutive_missing = 0  # valid measurement: reset gap counter

        if not self.initialized:
            self.x_estimate = log_measurement
            self.p_estimate = self.R
            self.initialized = True
            self._init_frame = len(self.history_x_log)
            self.history_measurements.append(area_measurement)
            self.history_x.append([area_measurement])
            self.history_x_log.append(log_measurement)
            self.history_p.append([self.p_estimate])
            self.history_p_pred.append(self.p_estimate)
            return area_measurement

        # Prediction step: grow uncertainty before incorporating measurement
        p_pred = self.p_estimate + self.Q
        S = p_pred + self.R

        if reject_outliers:
            innovation = abs(log_measurement - self.x_estimate)
            if innovation > outlier_threshold * math.sqrt(S):
                # Outlier rejected: accept prediction, skip measurement update
                self.p_estimate = p_pred
                self.history_measurements.append(area_measurement)
                filtered_area = math.exp(self.x_estimate)
                self.history_x.append([filtered_area])
                self.history_x_log.append(self.x_estimate)
                self.history_p.append([self.p_estimate])
                self.history_p_pred.append(p_pred)
                return filtered_area

        K = p_pred / S
        self.x_estimate = self.x_estimate + K * (log_measurement - self.x_estimate)
        self.p_estimate = (1 - K) * p_pred

        filtered_area = math.exp(self.x_estimate)
        self.history_measurements.append(area_measurement)
        self.history_x.append([filtered_area])
        self.history_x_log.append(self.x_estimate)
        self.history_p.append([self.p_estimate])
        self.history_p_pred.append(p_pred)

        return filtered_area

    def predict(self) -> float:
        """Predict next state in original space."""
        self.p_estimate = self.p_estimate + self.Q
        return math.exp(self.x_estimate)

    def rts_smooth(self) -> list[float]:
        """Rauch-Tung-Striebel backward smoothing in log-space.

        Only applied from the first initialized frame onward.
        Pre-initialization frames are left as 0.0 (no data).
        RTS gain C = P(k|k) / P(k+1|k) is guaranteed ≤ 1 within initialized region.
        """
        n = len(self.history_x_log)
        if n == 0 or self._init_frame < 0:
            return [0.0] * n

        x_smooth = list(self.history_x_log)
        p_smooth = [self.history_p[i][0] for i in range(n)]

        # Backward pass only over initialized frames
        start = self._init_frame
        for k in range(n - 2, start - 1, -1):
            p_pred_k = self.history_p_pred[k + 1]  # P(k+1|k) stored at step k+1
            if p_pred_k <= 0:
                continue
            C = min(self.history_p[k][0] / p_pred_k, 1.0)  # clamp for numerical safety
            x_smooth[k] = self.history_x_log[k] + C * (x_smooth[k + 1] - self.history_x_log[k])
            p_smooth[k] = self.history_p[k][0] + C**2 * (p_smooth[k + 1] - p_pred_k)

        # Pre-init frames: no data
        for k in range(start):
            x_smooth[k] = float('-inf')

        smoothed_areas = [math.exp(max(x, -100)) if x > -1e9 else 0.0 for x in x_smooth]
        self.history_x = [[a] for a in smoothed_areas]
        self.history_x_log = x_smooth
        self.history_p = [[p] for p in p_smooth]

        return smoothed_areas

    def apply_ema(self, smoothed_areas: list[float], alpha: float | None = None) -> tuple[list[float], list[float]]:
        """Apply Exponential Moving Average to RTS-smoothed areas in log-space.

        Operating in log-space keeps EMA consistent with the Kalman filter's
        log-normal assumption and allows consumption rate to be computed as
        a simple difference of EMA log values.

        Args:
            smoothed_areas: Output of rts_smooth() in original pixel² space.
            alpha: EMA factor (0, 1). Defaults to self.ema_alpha.
                   Lower = heavier smoothing (~19-frame half-life at alpha=0.1).

        Returns:
            (ema_areas, ema_log_areas): EMA-smoothed areas in original and log space.
        """
        a = alpha if alpha is not None else self.ema_alpha
        n = len(smoothed_areas)
        ema_log: list[float] = [0.0] * n
        ema_area: list[float] = [0.0] * n

        # Find first non-zero to initialise EMA
        init = next((i for i, v in enumerate(smoothed_areas) if v > 0), None)
        if init is None:
            self.history_ema_log = ema_log
            return ema_area, ema_log

        ema_log[init] = math.log(smoothed_areas[init])
        ema_area[init] = smoothed_areas[init]

        for t in range(init + 1, n):
            if smoothed_areas[t] > 0:
                log_val = math.log(smoothed_areas[t])
            else:
                # No measurement: hold previous EMA (same logic as Kalman missing-frame handling)
                log_val = ema_log[t - 1]
            ema_log[t] = a * log_val + (1 - a) * ema_log[t - 1]
            ema_area[t] = math.exp(ema_log[t])

        self.history_ema_log = ema_log
        return ema_area, ema_log


def create_kalman_filter(config: KalmanFilterConfig | None = None) -> LogNormalKalmanFilter:
    """Factory function to create a configured Kalman Filter.

    Args:
        config: KalmanFilterConfig instance. Uses defaults if None.

    Returns:
        Initialized LogNormalKalmanFilter
    """
    if config is None:
        config = KalmanFilterConfig()

    kf = LogNormalKalmanFilter(sigma_R=config.sigma_R, sigma_Q=config.sigma_Q)
    kf.max_missing_frames = config.max_missing_frames
    kf.ema_alpha = config.ema_alpha
    return kf
