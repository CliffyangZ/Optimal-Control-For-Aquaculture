"""Activity level mapping and threshold validation."""

from enum import Enum
from config.setting import ACTIVITY_THRESHOLDS


class ActivityLevel(str, Enum):
    """Activity level classification."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


def validate_thresholds(thresholds: dict) -> None:
    """Validate that thresholds cover 0-100 exactly once and are in correct order."""
    if not isinstance(thresholds, dict):
        raise ValueError("Thresholds must be a dictionary")

    if set(thresholds.keys()) != {"low", "medium", "high"}:
        raise ValueError("Thresholds must have exactly 'low', 'medium', and 'high' keys")

    ranges = []
    for level in ["low", "medium", "high"]:
        r = thresholds[level]
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            raise ValueError(f"Threshold range for '{level}' must be [min, max]")
        if not isinstance(r[0], int) or not isinstance(r[1], int):
            raise ValueError(f"Threshold values for '{level}' must be integers")
        if r[0] < 0 or r[1] > 100 or r[0] > r[1]:
            raise ValueError(f"Threshold range for '{level}' invalid: {r}")
        ranges.append(r)

    covered = set()
    for r in ranges:
        for score in range(r[0], r[1] + 1):
            if score in covered:
                raise ValueError(f"Score {score} covered by multiple thresholds")
            covered.add(score)

    if covered != set(range(0, 101)):
        raise ValueError("Thresholds do not cover all scores 0-100 exactly once")


def get_activity_level(score: int, thresholds: dict = None) -> ActivityLevel:
    """Map activity score (0-100) to activity level."""
    if thresholds is None:
        thresholds = ACTIVITY_THRESHOLDS

    if not isinstance(score, int) or score < 0 or score > 100:
        raise ValueError(f"Activity score must be an integer 0-100, got {score}")

    for level, (min_val, max_val) in thresholds.items():
        if min_val <= score <= max_val:
            return ActivityLevel[level.upper()]

    raise ValueError(f"Score {score} does not map to any activity level")


validate_thresholds(ACTIVITY_THRESHOLDS)
