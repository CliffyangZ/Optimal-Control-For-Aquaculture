"""Unit tests for spatial feature extraction functions."""

import pytest
import numpy as np
import cv2
from core.spatial_features import radial_edge_score, edge_density, gradient_consistency


class TestRadialEdgeScore:
    """Test radial_edge_score function with synthetic patterns."""

    def test_concentric_circles_high_score(self):
        """Edges in concentric circles should have high radial alignment."""
        # Create image with concentric circles
        img = np.zeros((200, 200), dtype=np.uint8)
        center = (100, 100)
        cv2.circle(img, center, 50, 255, 2)
        cv2.circle(img, center, 75, 255, 2)
        cv2.circle(img, center, 100, 255, 2)

        score = radial_edge_score(img, center[0], center[1], 20, 120, 50, 150)
        assert 0 <= score <= 1, "Score out of range"
        assert score > 0.5, "Concentric circles should have high radial score"

    def test_horizontal_lines_low_score(self):
        """Horizontal lines should have low radial alignment."""
        img = np.zeros((200, 200), dtype=np.uint8)
        for y in range(50, 150, 10):
            img[y, 40:160] = 255
        center = (100, 100)

        score = radial_edge_score(img, center[0], center[1], 10, 80, 50, 150)
        assert 0 <= score <= 1
        assert score < 0.7, "Horizontal lines should have lower radial score"

    def test_empty_image_returns_zero(self):
        """Empty image should return 0.0."""
        img = np.zeros((200, 200), dtype=np.uint8)
        score = radial_edge_score(img, 100, 100, 10, 80, 50, 150)
        assert score == 0.0


class TestEdgeDensity:
    """Test edge_density function."""

    def test_all_edges(self):
        """Cell with all pixels as edges should return 1.0."""
        edges = np.ones((100, 100), dtype=np.uint8) * 255
        cell_bounds = (10, 30, 10, 30)
        density = edge_density(edges, cell_bounds)
        assert density == 1.0

    def test_no_edges(self):
        """Cell with no edges should return 0.0."""
        edges = np.zeros((100, 100), dtype=np.uint8)
        cell_bounds = (10, 30, 10, 30)
        density = edge_density(edges, cell_bounds)
        assert density == 0.0

    def test_half_edges(self):
        """Cell with half edges should return ~0.5."""
        edges = np.zeros((100, 100), dtype=np.uint8)
        edges[10:30, 10:20] = 255  # Fill half the cell
        cell_bounds = (10, 30, 10, 30)
        density = edge_density(edges, cell_bounds)
        assert 0.45 < density < 0.55

    def test_empty_cell(self):
        """Empty cell bounds should return 0.0."""
        edges = np.ones((100, 100), dtype=np.uint8) * 255
        cell_bounds = (0, 0, 0, 0)
        density = edge_density(edges, cell_bounds)
        assert density == 0.0


class TestGradientConsistency:
    """Test gradient_consistency function."""

    def test_uniform_gradient_high_consistency(self):
        """Uniform gradient should have high consistency."""
        gx = np.ones((100, 100), dtype=np.float32) * 10.0
        gy = np.zeros((100, 100), dtype=np.float32)
        cell_bounds = (10, 30, 10, 30)

        consistency = gradient_consistency(gx, gy, cell_bounds)
        assert 0 <= consistency <= 1
        assert consistency > 0.9, "Uniform gradient should have high consistency"

    def test_random_gradient_low_consistency(self):
        """Random gradients should have low consistency."""
        np.random.seed(42)
        gx = np.random.randn(100, 100).astype(np.float32) * 10.0
        gy = np.random.randn(100, 100).astype(np.float32) * 10.0
        cell_bounds = (10, 30, 10, 30)

        consistency = gradient_consistency(gx, gy, cell_bounds)
        assert 0 <= consistency <= 1
        assert consistency < 0.5, "Random gradients should have low consistency"

    def test_zero_gradient_returns_zero(self):
        """Zero gradients should return 0.0."""
        gx = np.zeros((100, 100), dtype=np.float32)
        gy = np.zeros((100, 100), dtype=np.float32)
        cell_bounds = (10, 30, 10, 30)

        consistency = gradient_consistency(gx, gy, cell_bounds)
        assert consistency == 0.0
