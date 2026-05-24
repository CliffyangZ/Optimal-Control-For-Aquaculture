"""Kalman Filter for smoothing feed area estimates with outlier rejection and RTS smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class KalmanFilterConfig:
    """Configuration for Kalman Filter behavior."""

    sigma_R: float = 0.15
    """Measurement noise standard deviation (observational error)."""

    sigma_Q: float = 5e-3
    """Process noise standard deviation (model uncertainty)."""

    reject_outliers: bool = True
    """Whether to reject measurements >N*sigma from predicted value."""

    outlier_threshold: float = 3.0
    """Standard deviations threshold for outlier rejection."""

    use_log_space: bool = True
    """Whether to work in log-space (log-normal filter)."""


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
    This filter operates in log-space for better numerical stability.
    """

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
            self.history_measurements.append(0.0)
            self.history_x.append([math.exp(self.x_estimate)])
            self.history_p.append([self.p_estimate])
            return math.exp(self.x_estimate) if self.initialized else 0.0

        log_measurement = math.log(area_measurement)

        if not self.initialized:
            self.x_estimate = log_measurement
            self.p_estimate = self.R
            self.initialized = True
            self.history_measurements.append(area_measurement)
            self.history_x.append([area_measurement])
            self.history_p.append([self.p_estimate])
            return area_measurement

        predicted_log_y = self.x_estimate
        S = self.p_estimate + self.R

        if reject_outliers:
            innovation = abs(log_measurement - predicted_log_y)
            if innovation > outlier_threshold * math.sqrt(S):
                self.history_measurements.append(area_measurement)
                self.history_x.append([math.exp(self.x_estimate)])
                self.history_p.append([self.p_estimate])
                return math.exp(self.x_estimate)

        K = self.p_estimate / S
        self.x_estimate = self.x_estimate + K * (log_measurement - self.x_estimate)
        self.p_estimate = (1 - K) * self.p_estimate

        filtered_area = math.exp(self.x_estimate)
        self.history_measurements.append(area_measurement)
        self.history_x.append([filtered_area])
        self.history_p.append([self.p_estimate])

        return filtered_area

    def predict(self) -> float:
        """Predict next state in original space."""
        self.p_estimate = self.p_estimate + self.Q
        return math.exp(self.x_estimate)

    def rts_smooth(self) -> list[float]:
        """RTS backward smoothing in log-space."""
        n = len(self.history_x)
        if n == 0:
            return []

        x_log = [math.log(self.history_x[i][0]) if self.history_x[i][0] > 0 else 0.0 for i in range(n)]
        x_smooth = [[x_log[i]] for i in range(n)]
        p_smooth = [self.history_p[i][0] for i in range(n)]

        for k in range(n - 2, -1, -1):
            x_pred = x_log[k]
            p_pred = self.history_p[k][0] + self.Q
            C = self.history_p[k][0] / p_pred if p_pred > 0 else 0.0

            x_smooth[k][0] = x_log[k] + C * (x_smooth[k + 1][0] - x_pred)
            p_smooth[k] = self.history_p[k][0] + C**2 * (p_smooth[k + 1] - p_pred) if p_pred > 0 else self.history_p[k][0]

        smoothed_areas = [math.exp(x_smooth[i][0]) for i in range(n)]
        self.history_x = [[a] for a in smoothed_areas]
        self.history_p = [[p] for p in p_smooth]

        return smoothed_areas


def create_kalman_filter(config: KalmanFilterConfig | None = None) -> LogNormalKalmanFilter:
    """Factory function to create a configured Kalman Filter.

    Args:
        config: KalmanFilterConfig instance. Uses defaults if None.

    Returns:
        Initialized LogNormalKalmanFilter
    """
    if config is None:
        config = KalmanFilterConfig()

    return LogNormalKalmanFilter(sigma_R=config.sigma_R, sigma_Q=config.sigma_Q)
