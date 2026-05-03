"""YOLO-based pose estimation with skeleton keypoint coordinates."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


@dataclass(frozen=True)
class PoseKeypoint:
    """Single skeleton keypoint coordinate."""

    name: str
    x: int
    y: int
    confidence: float | None = None


@dataclass(frozen=True)
class PoseSkeleton:
    """Single detected person skeleton."""

    person_id: int
    box: tuple[int, int, int, int] | None
    confidence: float | None
    keypoints: list[PoseKeypoint]

    def coordinates(self) -> dict[str, dict[str, float | int | None]]:
        """Return keypoints as a name-indexed coordinate mapping."""
        return {
            keypoint.name: {
                "x": keypoint.x,
                "y": keypoint.y,
                "confidence": keypoint.confidence,
            }
            for keypoint in self.keypoints
        }


@dataclass(frozen=True)
class PoseEstimationResult:
    """Completed pose estimation result for one frame."""

    skeletons: list[PoseSkeleton]
    frame_sequence_number: int | None = None
    completed_at: datetime = field(default_factory=datetime.now)
    reason: str | None = None

    @property
    def available(self) -> bool:
        """Whether pose estimation completed without setup/runtime errors."""
        return self.reason is None


class YoloPoseEstimator:
    """Detect human pose keypoints with a YOLO pose model."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        keypoint_confidence_threshold: float = 0.25,
        yolo_factory: Any | None = None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        self._yolo_factory = yolo_factory
        self._model = None

    def estimate(self, frame) -> PoseEstimationResult:
        """Estimate pose skeletons in a CurrentFrame-compatible object."""
        image = getattr(frame, "image", None)
        sequence_number = getattr(frame, "sequence_number", None)
        if image is None or image.size == 0:
            return PoseEstimationResult(
                skeletons=[],
                frame_sequence_number=sequence_number,
                reason="Empty or invalid frame",
            )

        try:
            model = self._get_model()
            skeletons = self._estimate_skeletons(model, image)
        except Exception as exc:
            return PoseEstimationResult(
                skeletons=[],
                frame_sequence_number=sequence_number,
                reason=f"YOLO pose model unavailable: {exc}",
            )

        return PoseEstimationResult(
            skeletons=skeletons,
            frame_sequence_number=sequence_number,
        )

    def draw(self, image: np.ndarray, result: PoseEstimationResult) -> np.ndarray:
        """Draw skeleton keypoints and limbs on a frame copy."""
        display = image.copy()
        for skeleton in result.skeletons:
            points = {index: point for index, point in enumerate(skeleton.keypoints)}
            for start_index, end_index in COCO_SKELETON_EDGES:
                start = points.get(start_index)
                end = points.get(end_index)
                if not self._is_visible(start) or not self._is_visible(end):
                    continue
                cv2.line(display, (start.x, start.y), (end.x, end.y), (0, 255, 255), 2)

            for keypoint in skeleton.keypoints:
                if not self._is_visible(keypoint):
                    continue
                cv2.circle(display, (keypoint.x, keypoint.y), 4, (0, 255, 0), -1)

            if skeleton.box:
                x1, y1, x2, y2 = skeleton.box
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 128, 0), 2)
                label = f"Person {skeleton.person_id}"
                cv2.putText(
                    display,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 128, 0),
                    2,
                )
        return display

    def _get_model(self):
        if self._model is None:
            model_path = Path(self.model_path)
            if (
                self._yolo_factory is None
                and model_path.suffix == ".pt"
                and model_path.parent != Path(".")
            ):
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Model file not found at {self.model_path}. "
                        "Place the YOLOv26 pose weights there or set POSE_YOLO_MODEL_PATH."
                    )
            factory = self._yolo_factory
            if factory is None:
                from ultralytics import YOLO

                factory = YOLO
            self._model = factory(self.model_path)
        return self._model

    def _estimate_skeletons(self, model, image: np.ndarray) -> list[PoseSkeleton]:
        results = model.predict(image, conf=self.confidence_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        xy_points = self._as_array(getattr(getattr(result, "keypoints", None), "xy", None))
        if xy_points is None or xy_points.size == 0:
            return []

        keypoint_confidences = self._as_array(
            getattr(getattr(result, "keypoints", None), "conf", None)
        )
        boxes = getattr(result, "boxes", None)
        skeletons = []
        for person_index, person_points in enumerate(xy_points):
            keypoints = []
            for keypoint_index, point in enumerate(person_points):
                if keypoint_index >= len(COCO_KEYPOINT_NAMES):
                    break
                confidence = self._keypoint_confidence(
                    keypoint_confidences,
                    person_index,
                    keypoint_index,
                )
                keypoints.append(
                    PoseKeypoint(
                        name=COCO_KEYPOINT_NAMES[keypoint_index],
                        x=int(point[0]),
                        y=int(point[1]),
                        confidence=confidence,
                    )
                )

            box, confidence = self._person_box(boxes, person_index, image.shape[1], image.shape[0])
            skeletons.append(
                PoseSkeleton(
                    person_id=person_index + 1,
                    box=box,
                    confidence=confidence,
                    keypoints=keypoints,
                )
            )
        return skeletons

    def _person_box(self, boxes, person_index: int, width: int, height: int):
        if boxes is None or person_index >= len(boxes):
            return None, None
        box = boxes[person_index]
        xyxy = np.asarray(box.xyxy[0], dtype=float)
        confidence = float(np.asarray(box.conf[0], dtype=float))
        x1, y1, x2, y2 = xyxy.astype(int).tolist()
        return (
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        ), confidence

    def _keypoint_confidence(
        self,
        confidences: np.ndarray | None,
        person_index: int,
        keypoint_index: int,
    ) -> float | None:
        if confidences is None or confidences.size == 0:
            return None
        return float(confidences[person_index][keypoint_index])

    def _is_visible(self, keypoint: PoseKeypoint | None) -> bool:
        if keypoint is None:
            return False
        if keypoint.x <= 0 and keypoint.y <= 0:
            return False
        if keypoint.confidence is None:
            return True
        return keypoint.confidence >= self.keypoint_confidence_threshold

    def _as_array(self, value) -> np.ndarray | None:
        if value is None:
            return None
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)
