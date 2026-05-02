"""Tests for ripple detection and activity scoring."""

import pytest
import numpy as np
from datetime import datetime
from core.models import CurrentFrame, RippleDetectionResult, DetectionStatus
from core.ripple_detector import RippleDetector
from core.activity_levels import ActivityLevel


class TestRippleDetector:
    """Test ripple detection and activity scoring."""

    @pytest.fixture
    def detector(self):
        """Create a fresh ripple detector."""
        return RippleDetector()

    def create_test_frame(self, image_array: np.ndarray = None, seq_num: int = 1):
        """Helper to create a test frame."""
        if image_array is None:
            image_array = np.zeros((720, 1280, 3), dtype=np.uint8)

        return CurrentFrame(
            image=image_array,
            captured_at=datetime.now(),
            sequence_number=seq_num,
            source_width=1280,
            source_height=720
        )

    def test_detector_returns_detection_result(self, detector):
        """Test that detector returns a DetectionResult object."""
        frame = self.create_test_frame()
        result = detector.detect(frame)

        assert isinstance(result, RippleDetectionResult)

    def test_detection_result_has_status(self, detector):
        """Test that detection result includes status."""
        frame = self.create_test_frame()
        result = detector.detect(frame)

        assert result.status in [DetectionStatus.AVAILABLE, DetectionStatus.UNAVAILABLE]

    def test_available_detection_has_score_and_level(self, detector):
        """Test that available detection includes activity score and level."""
        frame = self.create_test_frame()
        result = detector.detect(frame)

        if result.status == DetectionStatus.AVAILABLE:
            assert result.activity_score is not None
            assert 0 <= result.activity_score <= 100
            assert result.activity_level in [
                ActivityLevel.LOW,
                ActivityLevel.MEDIUM,
                ActivityLevel.HIGH
            ]

    def test_unavailable_detection_has_reason(self, detector):
        """Test that unavailable detection includes a reason."""
        frame = self.create_test_frame()
        result = detector.detect(frame)

        if result.status == DetectionStatus.UNAVAILABLE:
            assert result.reason is not None
            assert isinstance(result.reason, str)
            assert result.activity_score is None
            assert result.activity_level is None

    def test_detection_includes_visual_marker(self, detector):
        """Test that detection result includes visual marker."""
        frame = self.create_test_frame()
        result = detector.detect(frame)

        assert hasattr(result, 'visual_marker')

    def test_detection_includes_frame_sequence(self, detector):
        """Test that detection includes the frame sequence number."""
        frame = self.create_test_frame(seq_num=42)
        result = detector.detect(frame)

        assert result.frame_sequence_number == 42

    def test_detection_timestamp(self, detector):
        """Test that detection includes completion timestamp."""
        frame = self.create_test_frame()
        result = detector.detect(frame)

        assert result.completed_at is not None
        assert isinstance(result.completed_at, datetime)

    def test_score_bounds(self, detector):
        """Test that activity scores are within 0-100 range."""
        dark_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        bright_frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
        random_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

        for frame_array in [dark_frame, bright_frame, random_frame]:
            frame = self.create_test_frame(frame_array)
            result = detector.detect(frame)

            if result.status == DetectionStatus.AVAILABLE:
                assert 0 <= result.activity_score <= 100

    def test_detection_reason_descriptive(self, detector):
        """Test that unavailable detection reasons are descriptive."""
        frame = self.create_test_frame()
        result = detector.detect(frame)

        if result.status == DetectionStatus.UNAVAILABLE:
            expected_reasons = {
                "Too dark",
                "Too bright",
                "Unstable scene",
                "Low confidence scene"
            }
            assert any(r in result.reason for r in expected_reasons)
