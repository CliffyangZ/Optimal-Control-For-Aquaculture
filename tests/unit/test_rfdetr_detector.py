"""Unit tests for RFDETRDetector result conversion."""

import numpy as np
import pytest

from core.infer.rfdetr_detector import RFDETRDetector


class _FakeDetections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.array(xyxy, dtype=float)
        self.confidence = np.array(confidence, dtype=float)
        self.class_id = np.array(class_id, dtype=int)

    def __len__(self):
        return len(self.xyxy)


class _FakeModel:
    class_names = ["fish"]

    def __init__(self, detections):
        self.detections = detections

    def predict(self, frame, threshold):
        return self.detections


def test_rfdetr_detector_clips_bbox_before_normalization() -> None:
    detections = _FakeDetections(
        xyxy=[[-0.2, -10, 220, 120]],
        confidence=[0.9],
        class_id=[0],
    )
    detector = RFDETRDetector(_FakeModel(detections), conf_threshold=0.25)

    detection = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8)).detections[0]

    assert detection.bbox_x == 0.0
    assert detection.bbox_y == 0.0
    assert detection.bbox_w == pytest.approx(1.0)
    assert detection.bbox_h == pytest.approx(1.0)
