"""Unit tests for activity score fusion and smoothing."""

import pytest
import numpy as np
from collections import deque
from config import setting


class TestWeightedFusion:
    """Test weighted combination of spatial and temporal features."""

    def test_all_zeros_gives_zero(self):
        """All-zero features should fuse to 0.0."""
        features = {
            'radial_edge_score': 0.0,
            'radial_expansion_score': 0.0,
            'motion_energy': 0.0,
            'edge_change_ratio': 0.0,
        }

        score = (
            setting.WEIGHT_SPATIAL * features['radial_edge_score'] +
            setting.WEIGHT_EXPANSION * features['radial_expansion_score'] +
            setting.WEIGHT_MOTION * features['motion_energy'] +
            setting.WEIGHT_EDGE_CHANGE * features['edge_change_ratio']
        )

        assert score == 0.0

    def test_all_ones_gives_one(self):
        """All-one features should fuse to 1.0."""
        features = {
            'radial_edge_score': 1.0,
            'radial_expansion_score': 1.0,
            'motion_energy': 1.0,
            'edge_change_ratio': 1.0,
        }

        score = (
            setting.WEIGHT_SPATIAL * features['radial_edge_score'] +
            setting.WEIGHT_EXPANSION * features['radial_expansion_score'] +
            setting.WEIGHT_MOTION * features['motion_energy'] +
            setting.WEIGHT_EDGE_CHANGE * features['edge_change_ratio']
        )

        assert abs(score - 1.0) < 1e-6

    def test_partial_features(self):
        """Partial features should fuse correctly."""
        features = {
            'radial_edge_score': 0.5,
            'radial_expansion_score': 1.0,
            'motion_energy': 0.0,
            'edge_change_ratio': 0.5,
        }

        score = (
            setting.WEIGHT_SPATIAL * features['radial_edge_score'] +
            setting.WEIGHT_EXPANSION * features['radial_expansion_score'] +
            setting.WEIGHT_MOTION * features['motion_energy'] +
            setting.WEIGHT_EDGE_CHANGE * features['edge_change_ratio']
        )

        # Manual calculation:
        # 0.25*0.5 + 0.35*1.0 + 0.25*0.0 + 0.15*0.5
        # = 0.125 + 0.35 + 0.0 + 0.075 = 0.55
        expected = 0.125 + 0.35 + 0.0 + 0.075
        assert abs(score - expected) < 1e-6

    def test_score_in_valid_range(self):
        """Fused score must be in [0, 1]."""
        for spatial in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for expansion in [0.0, 0.35, 0.7, 1.0]:
                for motion in [0.0, 0.25, 0.5, 1.0]:
                    for edge_change in [0.0, 0.15, 0.3, 1.0]:
                        score = (
                            setting.WEIGHT_SPATIAL * spatial +
                            setting.WEIGHT_EXPANSION * expansion +
                            setting.WEIGHT_MOTION * motion +
                            setting.WEIGHT_EDGE_CHANGE * edge_change
                        )
                        assert 0.0 <= score <= 1.0, \
                            f"Score {score} out of range for ({spatial}, {expansion}, {motion}, {edge_change})"


class TestSlidingWindowSmoothing:
    """Test sliding window mean for temporal smoothing."""

    def test_deque_mean_single_value(self):
        """Deque with single value should return that value."""
        window = deque([0.5], maxlen=30)
        mean = sum(window) / len(window)
        assert mean == 0.5

    def test_deque_mean_multiple_values(self):
        """Deque mean should be correct."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        window = deque(values, maxlen=10)
        mean = sum(window) / len(window)
        expected = sum(values) / len(values)
        assert abs(mean - expected) < 1e-6

    def test_deque_maxlen_enforcement(self):
        """Deque should drop old values when maxlen exceeded."""
        window = deque(maxlen=3)
        window.append(1.0)
        window.append(2.0)
        window.append(3.0)
        assert len(window) == 3

        window.append(4.0)
        assert len(window) == 3
        assert list(window) == [2.0, 3.0, 4.0]

    def test_window_convergence(self):
        """New values should converge toward constant value."""
        window = deque(maxlen=5)
        constant_value = 0.7

        for _ in range(10):
            window.append(constant_value)

        mean = sum(window) / len(window)
        assert abs(mean - constant_value) < 1e-6

    def test_ramp_then_constant(self):
        """Window behavior with ramp then constant values."""
        window = deque(maxlen=3)

        # Add ramp: 0.1, 0.2, 0.3
        values = [0.1, 0.2, 0.3, 0.5, 0.5, 0.5]
        results = []

        for val in values:
            window.append(val)
            mean = sum(window) / len(window)
            results.append(mean)

        # First three: means are 0.1, 0.15, 0.2
        assert abs(results[0] - 0.1) < 1e-6
        assert abs(results[1] - 0.15) < 1e-6
        assert abs(results[2] - 0.2) < 1e-6

        # Last three should converge to 0.5
        assert abs(results[3] - (0.2 + 0.3 + 0.5) / 3) < 1e-6
        assert abs(results[5] - 0.5) < 1e-6


class TestActivityLevelMapping:
    """Test activity level threshold mapping."""

    def test_low_threshold(self):
        """Score below LOW_THRESHOLD should map to 'low'."""
        score = 0.1
        level = "low" if score < setting.ACTIVITY_LOW_THRESHOLD else "medium"
        assert level == "low"

    def test_high_threshold(self):
        """Score >= HIGH_THRESHOLD should map to 'high'."""
        score = 0.6
        level = "high" if score >= setting.ACTIVITY_HIGH_THRESHOLD else "medium"
        assert level == "high"

    def test_medium_threshold(self):
        """Score between thresholds should map to 'medium'."""
        score = 0.35
        if score < setting.ACTIVITY_LOW_THRESHOLD:
            level = "low"
        elif score >= setting.ACTIVITY_HIGH_THRESHOLD:
            level = "high"
        else:
            level = "medium"
        assert level == "medium"

    def test_threshold_boundaries(self):
        """Test exact boundary values."""
        # Exactly at low threshold
        score = setting.ACTIVITY_LOW_THRESHOLD
        level = "low" if score < setting.ACTIVITY_LOW_THRESHOLD else "medium"
        assert level == "medium"  # >= LOW_THRESHOLD

        # Exactly at high threshold
        score = setting.ACTIVITY_HIGH_THRESHOLD
        level = "high" if score >= setting.ACTIVITY_HIGH_THRESHOLD else "medium"
        assert level == "high"
