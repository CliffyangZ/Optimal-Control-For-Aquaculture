"""Tests for Kalman Filter implementation."""

import math
import pytest

from core.infer.kalman_filter import (
    KalmanFilter,
    KalmanFilterConfig,
    LogNormalKalmanFilter,
    create_kalman_filter,
)


class TestKalmanFilter:
    """Tests for basic linear Kalman Filter."""

    def test_initialization(self):
        """Test filter initializes correctly."""
        kf = KalmanFilter(sigma_R=0.1, sigma_Q=0.01)
        assert kf.x_estimate == 0.0
        assert kf.p_estimate == 1.0
        assert not kf.initialized

    def test_first_measurement_initialization(self):
        """Test first measurement initializes the filter."""
        kf = KalmanFilter(sigma_R=0.1, sigma_Q=0.01)
        result = kf.update(10.0)

        assert kf.initialized
        assert kf.x_estimate == 10.0
        assert result == 10.0

    def test_zero_measurement_skipping(self):
        """Test that zero measurements are skipped."""
        kf = KalmanFilter(sigma_R=0.1, sigma_Q=0.01)
        kf.update(10.0)

        prev_estimate = kf.x_estimate
        kf.update(0.0)

        assert kf.x_estimate == prev_estimate

    def test_sequential_updates(self):
        """Test sequential updates converge estimate."""
        kf = KalmanFilter(sigma_R=0.1, sigma_Q=0.01)

        measurements = [100.0, 102.0, 101.5, 99.8, 101.2]
        estimates = []

        for m in measurements:
            est = kf.update(m)
            estimates.append(est)

        assert len(estimates) == 5
        assert estimates[1] != measurements[1]

    def test_outlier_rejection(self):
        """Test that outliers are rejected when enabled."""
        kf = KalmanFilter(sigma_R=0.1, sigma_Q=0.01)
        kf.update(100.0)

        normal_est = kf.update(102.0, reject_outliers=False)
        kf.x_estimate = 100.0
        kf.p_estimate = 1.0

        outlier_est = kf.update(500.0, reject_outliers=True, outlier_threshold=3.0)

        assert kf.x_estimate == 100.0

    def test_history_tracking(self):
        """Test that filter tracks history correctly."""
        kf = KalmanFilter(sigma_R=0.1, sigma_Q=0.01)

        measurements = [10.0, 12.0, 11.0]
        for m in measurements:
            kf.update(m)

        assert len(kf.history_x) == 3
        assert len(kf.history_measurements) == 3
        assert kf.history_measurements == measurements


class TestLogNormalKalmanFilter:
    """Tests for log-normal Kalman Filter."""

    def test_log_space_filtering(self):
        """Test filtering in log-space."""
        kf = LogNormalKalmanFilter(sigma_R=0.15, sigma_Q=5e-3)
        kf.update(1000.0)
        result = kf.update(1200.0)

        assert result > 0
        assert 1000 < result < 1200

    def test_zero_handling_in_log_space(self):
        """Test that zero measurements don't crash log-space filter."""
        kf = LogNormalKalmanFilter(sigma_R=0.15, sigma_Q=5e-3)
        kf.update(1000.0)
        result = kf.update(0.0)

        assert result > 0

    def test_log_normal_configuration(self):
        """Test LogNormalKalmanFilter with config."""
        config = KalmanFilterConfig(
            sigma_R=0.15,
            sigma_Q=5e-3,
            reject_outliers=True,
            outlier_threshold=3.0,
        )

        kf = create_kalman_filter(config)
        assert isinstance(kf, LogNormalKalmanFilter)
        assert kf.sigma_R == 0.15
        assert kf.sigma_Q == 5e-3

    def test_outlier_rejection_in_log_space(self):
        """Test outlier rejection works in log-space."""
        kf = LogNormalKalmanFilter(sigma_R=0.15, sigma_Q=5e-3)
        kf.update(1000.0)

        prev_estimate = kf.x_estimate
        kf.update(10000.0, reject_outliers=True, outlier_threshold=3.0)

        assert kf.x_estimate == prev_estimate


class TestRTSSmoothing:
    """Tests for Rauch-Tung-Striebel backward smoothing."""

    def test_rts_smooth_reduces_variance(self):
        """Test that RTS smoothing reduces variance."""
        kf = KalmanFilter(sigma_R=0.2, sigma_Q=0.01)

        noisy_measurements = [100.0, 120.0, 95.0, 110.0, 105.0]
        for m in noisy_measurements:
            kf.update(m)

        forward_estimates = [x[0] for x in kf.history_x]
        forward_variance = sum((x - sum(forward_estimates) / len(forward_estimates)) ** 2 for x in forward_estimates)

        kf.rts_smooth()
        smoothed_estimates = [x[0] for x in kf.history_x]
        smoothed_variance = sum((x - sum(smoothed_estimates) / len(smoothed_estimates)) ** 2 for x in smoothed_estimates)

        assert smoothed_variance <= forward_variance

    def test_rts_smooth_in_log_space(self):
        """Test RTS smoothing works in log-normal space."""
        kf = LogNormalKalmanFilter(sigma_R=0.15, sigma_Q=5e-3)

        measurements = [1000.0, 1500.0, 900.0, 1200.0, 1100.0]
        for m in measurements:
            kf.update(m)

        smoothed = kf.rts_smooth()

        assert len(smoothed) == 5
        assert all(x > 0 for x in smoothed)

    def test_empty_history_smooth(self):
        """Test RTS smoothing handles empty history."""
        kf = KalmanFilter(sigma_R=0.1, sigma_Q=0.01)
        result = kf.rts_smooth()
        assert result == []


class TestIntegration:
    """Integration tests for typical use cases."""

    def test_feed_area_filtering_scenario(self):
        """Test realistic feed area filtering scenario."""
        config = KalmanFilterConfig(
            sigma_R=0.15,
            sigma_Q=5e-3,
            reject_outliers=True,
            outlier_threshold=3.0,
        )

        kf = create_kalman_filter(config)

        feed_areas = [
            500000.0,
            600000.0,
            0.0,
            550000.0,
            700000.0,
            4500000.0,
            400000.0,
            500000.0,
        ]

        results = []
        for area in feed_areas:
            est = kf.update(area, reject_outliers=config.reject_outliers, outlier_threshold=config.outlier_threshold)
            results.append(est)

        assert len(results) == len(feed_areas)
        assert all(x >= 0 for x in results)

    def test_predict_without_measurement(self):
        """Test prediction without measurement updates."""
        kf = KalmanFilter(sigma_R=0.1, sigma_Q=0.01)
        kf.update(100.0)

        prediction = kf.predict()
        assert prediction == 100.0

    def test_combined_zero_and_outlier_handling(self):
        """Test handling of both zeros and outliers."""
        kf = LogNormalKalmanFilter(sigma_R=0.15, sigma_Q=5e-3)

        measurements = [1000.0, 0.0, 1100.0, 500000.0, 950.0]

        for m in measurements:
            kf.update(m, reject_outliers=True, outlier_threshold=3.0)

        assert kf.initialized
        assert len(kf.history_x) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
