"""Tests for YOLO pose estimation."""

from datetime import datetime

import numpy as np

from core.models import CurrentFrame
from core.pose_estimation import PoseEstimationResult, YoloPoseEstimator


class FakeBox:
    """Minimal YOLO box object for tests."""

    def __init__(self, xyxy, confidence):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [np.array(confidence, dtype=np.float32)]


class FakeKeypoints:
    """Minimal YOLO keypoints object for tests."""

    def __init__(self):
        points = np.zeros((1, 17, 2), dtype=np.float32)
        for index in range(17):
            points[0, index] = [10 + index, 20 + index]
        self.xy = points
        self.conf = np.full((1, 17), 0.9, dtype=np.float32)


class FakeResult:
    """Minimal YOLO result object for tests."""

    def __init__(self):
        self.boxes = [FakeBox([5, 10, 90, 110], 0.88)]
        self.keypoints = FakeKeypoints()


class FakeYolo:
    """Deterministic YOLO pose model for tests."""

    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, image, conf, verbose):
        return [FakeResult()]


def create_frame(image):
    """Create a CurrentFrame for pose tests."""
    height = image.shape[0] if len(image.shape) > 0 else 0
    width = image.shape[1] if len(image.shape) > 1 else 0
    return CurrentFrame(
        image=image,
        captured_at=datetime.now(),
        sequence_number=7,
        source_width=width,
        source_height=height,
    )


def test_estimate_returns_pose_skeleton():
    """Estimator returns one skeleton with COCO keypoint coordinates."""
    estimator = YoloPoseEstimator(
        model_path="models/yolo-v26-pose.pt",
        yolo_factory=FakeYolo,
    )
    frame = create_frame(np.full((120, 160, 3), 120, dtype=np.uint8))

    result = estimator.estimate(frame)

    assert isinstance(result, PoseEstimationResult)
    assert result.available
    assert result.frame_sequence_number == 7
    assert len(result.skeletons) == 1
    assert result.skeletons[0].box == (5, 10, 90, 110)
    assert abs(result.skeletons[0].confidence - 0.88) < 0.001
    assert len(result.skeletons[0].keypoints) == 17
    assert result.skeletons[0].coordinates()["nose"] == {
        "x": 10,
        "y": 20,
        "confidence": 0.8999999761581421,
    }


def test_empty_frame_returns_unavailable():
    """Empty frames do not call YOLO and return a reason."""
    estimator = YoloPoseEstimator(
        model_path="models/yolo-v26-pose.pt",
        yolo_factory=FakeYolo,
    )
    frame = create_frame(np.array([], dtype=np.uint8))

    result = estimator.estimate(frame)

    assert not result.available
    assert result.reason == "Empty or invalid frame"
    assert result.skeletons == []


def test_missing_model_returns_setup_reason(tmp_path):
    """Missing configured model paths return an actionable reason."""
    estimator = YoloPoseEstimator(
        model_path=str(tmp_path / "models" / "missing-pose.pt"),
        yolo_factory=None,
    )
    frame = create_frame(np.full((120, 160, 3), 120, dtype=np.uint8))

    result = estimator.estimate(frame)

    assert not result.available
    assert "Model file not found" in result.reason
    assert "POSE_YOLO_MODEL_PATH" in result.reason


def test_draw_adds_skeleton_overlay():
    """Processed display draws skeleton overlay on a frame copy."""
    estimator = YoloPoseEstimator(
        model_path="models/yolo-v26-pose.pt",
        yolo_factory=FakeYolo,
    )
    image = np.full((120, 160, 3), 120, dtype=np.uint8)
    result = estimator.estimate(create_frame(image))

    display = estimator.draw(image, result)

    assert display.shape == image.shape
    assert not np.array_equal(display, image)
