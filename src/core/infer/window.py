"""DetectionWindow: sliding buffer of recent DetectionResults for feeding decisions.

The window holds the most recent `max_frames` DetectionResults in a FIFO deque.
Computed properties smooth per-frame noise into signals the FeedingStateMachine
can act on.
"""

from __future__ import annotations

from collections import deque

from core.types import DetectionResult


class DetectionWindow:
    """Sliding FIFO buffer of per-frame DetectionResults."""

    def __init__(self, max_frames: int = 30) -> None:
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        self._frames: deque[DetectionResult] = deque(maxlen=max_frames)
        self.max_frames = max_frames

    def push(self, result: DetectionResult) -> None:
        """Append a new DetectionResult; oldest frame auto-evicted when at capacity."""
        self._frames.append(result)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def mean_feed_count(self) -> float:
        """Mean number of `feed` detections per frame in the window."""
        if not self._frames:
            return 0.0
        total = sum(
            sum(1 for d in f.detections if d.class_name == "feed")
            for f in self._frames
        )
        return total / len(self._frames)

    @property
    def mean_debris_count(self) -> float:
        """Mean number of `debris` detections per frame in the window."""
        if not self._frames:
            return 0.0
        total = sum(
            sum(1 for d in f.detections if d.class_name == "debris")
            for f in self._frames
        )
        return total / len(self._frames)

    @property
    def is_ready(self) -> bool:
        """True once the window holds at least half its capacity of frames.

        Callers can gate feeding decisions on this to avoid acting on a single
        noisy frame at cold-start.
        """
        return len(self._frames) >= max(1, self.max_frames // 2)

    @property
    def fish_presence(self) -> bool:
        """True if any `fish` or `lumpfish` was detected in any frame in the window."""
        return any(
            any(d.class_name in ("fish", "lumpfish") for d in f.detections)
            for f in self._frames
        )
