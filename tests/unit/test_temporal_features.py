"""Unit tests for temporal feature extraction functions."""

import pytest
import numpy as np
import cv2
from core.temporal_features import (
    compute_flow, motion_energy, edge_change_ratio, radial_expansion_score
)


class TestMotionEnergy:
    """Test motion_energy function."""

    def test_identical_frames_zero_energy(self):
        """Identical frames should return 0.0."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        energy = motion_energy(gray, gray, threshold=12)
        assert energy == 0.0

    def test_completely_different_frames_high_energy(self):
        """Completely different frames should have high energy."""
        gray1 = np.ones((100, 100), dtype=np.uint8) * 0
        gray2 = np.ones((100, 100), dtype=np.uint8) * 255
        energy = motion_energy(gray1, gray2, threshold=12)
        assert energy == 1.0

    def test_partial_change(self):
        """Partial frame change should return partial energy."""
        gray1 = np.ones((100, 100), dtype=np.uint8) * 100
        gray2 = gray1.copy()
        gray2[0:50, 0:100] = 150  # Change top half
        energy = motion_energy(gray1, gray2, threshold=12)
        assert 0.4 < energy < 0.6, f"Expected ~0.5, got {energy}"

    def test_threshold_effect(self):
        """Energy should be zero below threshold."""
        gray1 = np.ones((100, 100), dtype=np.uint8) * 100
        gray2 = np.ones((100, 100), dtype=np.uint8) * 105  # Diff = 5
        energy = motion_energy(gray1, gray2, threshold=12)
        assert energy == 0.0

    def test_shape_mismatch(self):
        """Different shapes should return 0.0."""
        gray1 = np.ones((100, 100), dtype=np.uint8)
        gray2 = np.ones((100, 101), dtype=np.uint8)
        energy = motion_energy(gray1, gray2, threshold=12)
        assert energy == 0.0


class TestEdgeChangeRatio:
    """Test edge_change_ratio function."""

    def test_identical_frames_zero_change(self):
        """Identical frames should have zero edge change."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[40:60, 40:60] = 200
        ratio = edge_change_ratio(gray, gray, canny_low=50, canny_high=150)
        assert ratio == 0.0

    def test_completely_different_frames(self):
        """Completely different frames may have different edges."""
        gray1 = np.zeros((100, 100), dtype=np.uint8)
        gray1[40:60, 40:60] = 200
        gray2 = np.zeros((100, 100), dtype=np.uint8)
        gray2[10:30, 10:30] = 200

        ratio = edge_change_ratio(gray1, gray2, canny_low=50, canny_high=150)
        assert 0 <= ratio <= 1
        # With completely different patterns, most edges should change
        assert ratio > 0.5

    def test_no_edges_both_frames(self):
        """If no edges in either frame, ratio should be 0.0."""
        gray1 = np.ones((100, 100), dtype=np.uint8) * 128
        gray2 = np.ones((100, 100), dtype=np.uint8) * 130
        ratio = edge_change_ratio(gray1, gray2, canny_low=50, canny_high=150)
        assert ratio == 0.0


class TestRadialExpansionScore:
    """Test radial_expansion_score function."""

    def test_outward_flow_high_score(self):
        """Outward flow should give high expansion score."""
        h, w = 100, 100
        flow = np.zeros((h, w, 2), dtype=np.float32)

        # Create outward flow from center
        center_x, center_y = 50, 50
        for y in range(h):
            for x in range(w):
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx**2 + dy**2) + 1e-6
                flow[y, x, 0] = dx / dist * 2.0  # Outward x
                flow[y, x, 1] = dy / dist * 2.0  # Outward y

        score = radial_expansion_score(
            flow, center_x, center_y, r_min=5, r_max=40,
            mag_threshold=0.5, step=5
        )
        assert 0 <= score <= 1
        assert score > 0.7, f"Outward flow should have high score, got {score}"

    def test_inward_flow_low_score(self):
        """Inward flow should give low expansion score."""
        h, w = 100, 100
        flow = np.zeros((h, w, 2), dtype=np.float32)

        # Create inward flow toward center
        center_x, center_y = 50, 50
        for y in range(h):
            for x in range(w):
                dx = center_x - x  # Toward center
                dy = center_y - y
                dist = np.sqrt(dx**2 + dy**2) + 1e-6
                flow[y, x, 0] = dx / dist * 2.0
                flow[y, x, 1] = dy / dist * 2.0

        score = radial_expansion_score(
            flow, center_x, center_y, r_min=5, r_max=40,
            mag_threshold=0.5, step=5
        )
        assert 0 <= score <= 1
        assert score < 0.3, f"Inward flow should have low score, got {score}"

    def test_zero_flow_returns_zero(self):
        """Zero flow should return 0.0."""
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        score = radial_expansion_score(
            flow, 50, 50, r_min=5, r_max=40,
            mag_threshold=0.5, step=5
        )
        assert score == 0.0

    def test_threshold_filtering(self):
        """Very small magnitude flow below threshold should be filtered."""
        h, w = 100, 100
        flow = np.ones((h, w, 2), dtype=np.float32) * 0.01  # Very small

        score = radial_expansion_score(
            flow, 50, 50, r_min=5, r_max=40,
            mag_threshold=1.0, step=5
        )
        assert score == 0.0


class TestComputeFlow:
    """Test compute_flow function."""

    def test_identical_frames_zero_flow(self):
        """Identical frames should produce near-zero optical flow."""
        gray = np.ones((50, 50), dtype=np.uint8) * 128
        flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 21,
            'iterations': 3,
            'polyN': 5,
            'polySigma': 1.2,
        }

        flow = compute_flow(gray, gray, flow_params)

        assert flow.shape == (50, 50, 2)
        # Flow magnitude should be very small for identical frames
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_mag = np.mean(magnitude)
        assert mean_mag < 0.5, f"Identical frames should have near-zero flow, got {mean_mag}"

    def test_shifted_pattern_detects_motion(self):
        """Shifted pattern should produce non-zero optical flow."""
        # Create a small square pattern
        gray1 = np.zeros((50, 50), dtype=np.uint8)
        gray1[15:25, 15:25] = 255

        # Shift pattern to the right
        gray2 = np.zeros((50, 50), dtype=np.uint8)
        gray2[15:25, 18:28] = 255

        flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 21,
            'iterations': 3,
            'polyN': 5,
            'polySigma': 1.2,
        }

        flow = compute_flow(gray1, gray2, flow_params)

        assert flow.shape == (50, 50, 2)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_mag = np.mean(magnitude)
        assert mean_mag > 0.1, f"Shifted pattern should have detectable flow, got {mean_mag}"
