"""OpenCV-based ripple detection with explainable scoring."""

import cv2
import numpy as np
from datetime import datetime
from core.models import CurrentFrame, RippleDetectionResult, DetectionStatus
from core.activity_levels import get_activity_level, ActivityLevel


class RippleDetector:
    """Detects ripples using OpenCV analysis with confidence evaluation."""

    def __init__(self):
        self.brightness_threshold_low = 30
        self.brightness_threshold_high = 220
        self.stability_threshold = 0.85

    def detect(self, frame: CurrentFrame) -> RippleDetectionResult:
        """Analyze frame for ripple activity and confidence."""
        image = frame.image

        if image is None or image.size == 0:
            return RippleDetectionResult(
                status=DetectionStatus.UNAVAILABLE,
                reason="Empty or invalid frame",
                frame_sequence_number=frame.sequence_number,
                completed_at=datetime.now()
            )

        confidence_info = self._check_confidence(image)
        if not confidence_info["is_confident"]:
            return RippleDetectionResult(
                status=DetectionStatus.UNAVAILABLE,
                reason=confidence_info["reason"],
                confidence=confidence_info.get("confidence", 0),
                frame_sequence_number=frame.sequence_number,
                completed_at=datetime.now()
            )

        activity_score = self._calculate_ripple_score(image)
        activity_level = get_activity_level(activity_score)
        visual_marker = self._generate_visual_marker(image, activity_score)

        return RippleDetectionResult(
            status=DetectionStatus.AVAILABLE,
            activity_score=activity_score,
            activity_level=activity_level,
            confidence=confidence_info.get("confidence", 1.0),
            visual_marker=visual_marker,
            frame_sequence_number=frame.sequence_number,
            completed_at=datetime.now()
        )

    def _check_confidence(self, image: np.ndarray) -> dict:
        """Check if scene conditions allow reliable detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mean_brightness = np.mean(gray)

        if mean_brightness < self.brightness_threshold_low:
            return {
                "is_confident": False,
                "reason": "Too dark",
                "confidence": 0.0
            }

        if mean_brightness > self.brightness_threshold_high:
            return {
                "is_confident": False,
                "reason": "Too bright",
                "confidence": 0.0
            }

        variance = np.var(gray)
        if variance < 10:
            return {
                "is_confident": False,
                "reason": "Unstable scene",
                "confidence": 0.2
            }

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return {
                "is_confident": False,
                "reason": "Low confidence scene",
                "confidence": 0.3
            }

        return {
            "is_confident": True,
            "confidence": min(1.0, (mean_brightness / 128) * (variance / 100))
        }

    def _calculate_ripple_score(self, image: np.ndarray) -> int:
        """Calculate 0-100 activity score based on motion and texture cues."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size

        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        sharpness = np.std(laplacian)

        texture_score = min(100, int((edge_density * 100) + (sharpness / 2)))

        variation_score = int(np.std(gray) / 2.55)

        ripple_score = int(0.6 * texture_score + 0.4 * variation_score)
        return min(100, max(0, ripple_score))

    def _generate_visual_marker(self, image: np.ndarray, score: int) -> dict:
        """Generate visual marker metadata for processed display."""
        if score < 33:
            color = (0, 255, 0)
            label = "Low Activity"
        elif score < 67:
            color = (0, 165, 255)
            label = "Medium Activity"
        else:
            color = (0, 0, 255)
            label = "High Activity"

        return {
            "color": color,
            "label": label,
            "score": score,
            "mark_regions": self._get_activity_regions(image, score)
        }

    def _get_activity_regions(self, image: np.ndarray, score: int) -> list:
        """Identify regions of detected activity."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours[:10]:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

        return regions
