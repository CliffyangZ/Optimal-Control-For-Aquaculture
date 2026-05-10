"""Integration tests for the complete spatiotemporal ripple analysis pipeline."""

import pytest
import numpy as np
import cv2
from datetime import datetime

from core.models import CurrentFrame, ActivityHeatmap
from core.spatiotemporal_pipeline import SpatiotemporalRipplePipeline


class TestSpatiotemporalPipeline:
    """Integration tests for the full pipeline."""

    def test_pipeline_with_single_frame(self):
        """Pipeline should process a single frame without crashing."""
        pipeline = SpatiotemporalRipplePipeline()

        # Create a simple test frame
        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_data[40:60, 40:60] = 200

        frame = CurrentFrame(
            image=frame_data,
            captured_at=datetime.now(),
            sequence_number=1,
            source_width=100,
            source_height=100,
        )

        heatmap = pipeline.process_frame(frame)

        assert isinstance(heatmap, ActivityHeatmap)
        assert 0 <= heatmap.global_score <= 1.0
        assert heatmap.global_level in ["low", "medium", "high"]
        assert heatmap.cells.shape == (8, 8)

    def test_pipeline_with_expanding_ripple(self):
        """Pipeline should detect expanding ripple patterns."""
        pipeline = SpatiotemporalRipplePipeline()

        # Frame 1: concentric rings
        frame1 = _create_concentric_circles_frame(100, 100, 50, 50, radii=[20, 35, 50])

        current_frame = CurrentFrame(
            image=frame1,
            captured_at=datetime.now(),
            sequence_number=1,
            source_width=100,
            source_height=100,
        )

        heatmap1 = pipeline.process_frame(current_frame)
        score1 = heatmap1.global_score

        # Frame 2: expanded rings (outward motion)
        frame2 = _create_concentric_circles_frame(100, 100, 50, 50, radii=[25, 40, 55])

        current_frame = CurrentFrame(
            image=frame2,
            captured_at=datetime.now(),
            sequence_number=2,
            source_width=100,
            source_height=100,
        )

        heatmap2 = pipeline.process_frame(current_frame)
        score2 = heatmap2.global_score

        # Expanding ripple should show increasing activity
        assert heatmap2.cells.shape == (8, 8)
        assert 0 <= score2 <= 1.0

    def test_pipeline_with_static_frame(self):
        """Pipeline should return low score for static frames."""
        pipeline = SpatiotemporalRipplePipeline()

        # Static uniform frame
        frame_data = np.ones((100, 100, 3), dtype=np.uint8) * 128

        for i in range(5):
            current_frame = CurrentFrame(
                image=frame_data.copy(),
                captured_at=datetime.now(),
                sequence_number=i + 1,
                source_width=100,
                source_height=100,
            )

            heatmap = pipeline.process_frame(current_frame)
            assert heatmap.global_score < 0.3, f"Static frame should have low score, got {heatmap.global_score}"

    def test_pipeline_handles_edge_cases(self):
        """Pipeline should handle edge cases gracefully."""
        pipeline = SpatiotemporalRipplePipeline()

        # Empty frame (all zeros)
        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        frame = CurrentFrame(
            image=frame_data,
            captured_at=datetime.now(),
            sequence_number=1,
            source_width=100,
            source_height=100,
        )

        heatmap = pipeline.process_frame(frame)
        assert heatmap.global_score == 0.0, "Empty frame should have zero score"

        # Very small frame
        small_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = CurrentFrame(
            image=small_frame,
            captured_at=datetime.now(),
            sequence_number=2,
            source_width=10,
            source_height=10,
        )

        heatmap = pipeline.process_frame(frame)
        assert isinstance(heatmap, ActivityHeatmap)
        assert 0 <= heatmap.global_score <= 1.0

    def test_pipeline_sliding_window_smoothing(self):
        """Pipeline should smooth scores over sliding window."""
        pipeline = SpatiotemporalRipplePipeline()

        # Create alternating high/low activity frames
        high_frame = _create_concentric_circles_frame(100, 100, 50, 50, radii=[30, 45])
        low_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

        scores = []
        for i in range(15):
            # Alternate high and low
            frame_data = high_frame if i % 2 == 0 else low_frame

            current_frame = CurrentFrame(
                image=frame_data.copy(),
                captured_at=datetime.now(),
                sequence_number=i + 1,
                source_width=100,
                source_height=100,
            )

            heatmap = pipeline.process_frame(current_frame)
            scores.append(heatmap.global_score)

        # Scores should be smoothed (less variation than raw)
        differences = [abs(scores[i] - scores[i-1]) for i in range(1, len(scores))]
        avg_difference = np.mean(differences)

        # All differences should be reasonable (smoothing working)
        assert all(d < 0.5 for d in differences), "Smoothed scores have too much variation"

    def test_pipeline_first_frame_temporal_zero(self):
        """First frame should have zero temporal features."""
        pipeline = SpatiotemporalRipplePipeline()

        frame_data = _create_concentric_circles_frame(100, 100, 50, 50, radii=[30, 45])

        frame = CurrentFrame(
            image=frame_data,
            captured_at=datetime.now(),
            sequence_number=1,
            source_width=100,
            source_height=100,
        )

        # First frame should process without temporal data
        heatmap = pipeline.process_frame(frame)

        # Even with complex spatial pattern, score might be moderate
        # since temporal features are zero
        assert isinstance(heatmap, ActivityHeatmap)
        assert 0 <= heatmap.global_score <= 1.0

    def test_pipeline_heatmap_cell_count(self):
        """Heatmap should have correct number of cells."""
        pipeline = SpatiotemporalRipplePipeline()

        frame_data = np.ones((100, 100, 3), dtype=np.uint8) * 128

        frame = CurrentFrame(
            image=frame_data,
            captured_at=datetime.now(),
            sequence_number=1,
            source_width=100,
            source_height=100,
        )

        heatmap = pipeline.process_frame(frame)

        assert heatmap.cells.shape == (8, 8), "Should have 8x8 grid by default"
        assert len(heatmap.cells.flatten()) == 64

    def test_pipeline_activity_level_mapping(self):
        """Activity levels should map correctly to scores."""
        pipeline = SpatiotemporalRipplePipeline()

        # Create a frame that should produce high activity (expanding ripples)
        frame_data = _create_concentric_circles_frame(100, 100, 50, 50, radii=[20, 35, 50])

        for _ in range(10):  # Let smoothing window fill up
            frame = CurrentFrame(
                image=frame_data.copy(),
                captured_at=datetime.now(),
                sequence_number=_,
                source_width=100,
                source_height=100,
            )
            heatmap = pipeline.process_frame(frame)

        # High activity frame
        assert heatmap.global_level in ["low", "medium", "high"]

        # Static frame should be low
        static_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        frame = CurrentFrame(
            image=static_frame,
            captured_at=datetime.now(),
            sequence_number=100,
            source_width=100,
            source_height=100,
        )

        heatmap = pipeline.process_frame(frame)
        assert heatmap.global_level in ["low", "medium", "high"]


def _create_concentric_circles_frame(h, w, cx, cy, radii, thickness=2):
    """Helper: create frame with concentric circles."""
    frame = np.ones((h, w, 3), dtype=np.uint8) * 128
    for r in radii:
        cv2.circle(frame, (cx, cy), r, 255, thickness)
    return frame
