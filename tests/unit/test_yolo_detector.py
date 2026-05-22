"""Unit tests for YOLODetector result conversion."""

import numpy as np

from core.infer.detector import YOLODetector


class _FakeValue:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value, dtype=float)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return _FakeValue(self.value)


class _FakeBox:
    def __init__(self, xyxy, conf, cls_idx):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls_idx)


class _FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeModel:
    def __init__(self, result):
        self.result = result

    def __call__(self, raw_frame, conf, verbose):
        return [self.result]


def test_detector_preserves_unsupported_class_for_worker_validation() -> None:
    result = _FakeResult(
        boxes=[_FakeBox([10, 20, 30, 40], 0.9, 0)],
        names={0: "shrimp"},
    )
    detector = YOLODetector(_FakeModel(result), conf_threshold=0.25)

    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8)).detections

    assert len(detections) == 1
    assert detections[0].class_name == "shrimp"


def test_detector_suppresses_low_confidence_detections() -> None:
    result = _FakeResult(
        boxes=[_FakeBox([10, 20, 30, 40], 0.2, 0)],
        names={0: "fish"},
    )
    detector = YOLODetector(_FakeModel(result), conf_threshold=0.25)

    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8)).detections

    assert detections == []


def test_detector_clips_bbox_before_normalization() -> None:
    result = _FakeResult(
        boxes=[_FakeBox([-20, -10, 60, 50], 0.9, 0)],
        names={0: "fish"},
    )
    detector = YOLODetector(_FakeModel(result), conf_threshold=0.25)

    detection = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8)).detections[0]

    assert detection.bbox_x == 0.0
    assert detection.bbox_y == 0.0
    assert detection.bbox_w == 0.3
    assert detection.bbox_h == 0.5
