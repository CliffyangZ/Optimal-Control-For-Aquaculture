"""Unit tests for `core.infer.window.DetectionWindow`.

Covers:
- FIFO eviction when pushed beyond max_frames
- mean_feed_count with mixed feed/non-feed frames
- fish_presence true for lumpfish
- all-zero window edge case
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.infer.window import DetectionWindow
from core.types import Detection, DetectionResult


def _make_result(detections: list[tuple[str, int]] | None = None) -> DetectionResult:
    """Build a DetectionResult with the given (class_name, class_index) detections."""
    dets = []
    for name, idx in (detections or []):
        dets.append(Detection(
            class_name=name,
            class_index=idx,
            confidence=0.9,
            bbox_x=0.1,
            bbox_y=0.1,
            bbox_w=0.1,
            bbox_h=0.1,
        ))
    return DetectionResult(
        frame_id=0,
        captured_at=datetime.now(timezone.utc),
        source_width=640,
        source_height=480,
        detections=dets,
        inference_ms=10.0,
    )


def _fish_frame() -> DetectionResult:
    return _make_result([("fish", 2)])


def _feed_frame() -> DetectionResult:
    return _make_result([("feed", 1)])


def _lumpfish_frame() -> DetectionResult:
    return _make_result([("lumpfish", 3)])


def _debris_frame() -> DetectionResult:
    return _make_result([("debris", 0)])


def _empty_frame() -> DetectionResult:
    return _make_result()


# ---------------------------------------------------------------------------
# Basic push and length
# ---------------------------------------------------------------------------

def test_initial_length_is_zero() -> None:
    w = DetectionWindow(max_frames=5)
    assert len(w) == 0


def test_length_grows_with_pushes() -> None:
    w = DetectionWindow(max_frames=5)
    for _ in range(3):
        w.push(_empty_frame())
    assert len(w) == 3


# ---------------------------------------------------------------------------
# FIFO eviction
# ---------------------------------------------------------------------------

def test_fifo_eviction_when_beyond_max_frames() -> None:
    w = DetectionWindow(max_frames=3)
    # Push 3 empty frames, then one feed frame
    for _ in range(3):
        w.push(_empty_frame())
    assert len(w) == 3
    w.push(_feed_frame())
    # Should still be 3 (oldest evicted)
    assert len(w) == 3


def test_mean_feed_count_after_eviction() -> None:
    """After evicting all empty frames, feed frames should dominate."""
    w = DetectionWindow(max_frames=2)
    w.push(_empty_frame())
    w.push(_feed_frame())
    w.push(_feed_frame())
    # Window now: [feed, feed] — first empty was evicted
    assert w.mean_feed_count == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mean_feed_count
# ---------------------------------------------------------------------------

def test_mean_feed_count_empty_window() -> None:
    w = DetectionWindow(max_frames=10)
    assert w.mean_feed_count == pytest.approx(0.0)


def test_mean_feed_count_no_feed_detections() -> None:
    w = DetectionWindow(max_frames=5)
    w.push(_fish_frame())
    w.push(_debris_frame())
    assert w.mean_feed_count == pytest.approx(0.0)


def test_mean_feed_count_all_feed_frames() -> None:
    w = DetectionWindow(max_frames=4)
    for _ in range(4):
        w.push(_feed_frame())
    assert w.mean_feed_count == pytest.approx(1.0)


def test_mean_feed_count_mixed_frames() -> None:
    w = DetectionWindow(max_frames=4)
    w.push(_feed_frame())   # 1 feed
    w.push(_empty_frame())  # 0 feed
    w.push(_feed_frame())   # 1 feed
    w.push(_empty_frame())  # 0 feed
    assert w.mean_feed_count == pytest.approx(0.5)


def test_mean_feed_count_multiple_feed_per_frame() -> None:
    """A single frame with 3 feed detections should count as 3."""
    result = _make_result([("feed", 1), ("feed", 1), ("feed", 1)])
    w = DetectionWindow(max_frames=2)
    w.push(result)
    w.push(_empty_frame())
    assert w.mean_feed_count == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# mean_debris_count
# ---------------------------------------------------------------------------

def test_mean_debris_count_empty_window() -> None:
    w = DetectionWindow(max_frames=5)
    assert w.mean_debris_count == pytest.approx(0.0)


def test_mean_debris_count_mixed() -> None:
    w = DetectionWindow(max_frames=4)
    w.push(_debris_frame())  # 1 debris
    w.push(_empty_frame())   # 0
    w.push(_empty_frame())   # 0
    w.push(_empty_frame())   # 0
    assert w.mean_debris_count == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# fish_presence
# ---------------------------------------------------------------------------

def test_fish_presence_false_for_empty_window() -> None:
    w = DetectionWindow(max_frames=5)
    assert w.fish_presence is False


def test_fish_presence_true_for_fish() -> None:
    w = DetectionWindow(max_frames=5)
    w.push(_fish_frame())
    assert w.fish_presence is True


def test_fish_presence_true_for_lumpfish() -> None:
    w = DetectionWindow(max_frames=5)
    w.push(_lumpfish_frame())
    assert w.fish_presence is True


def test_fish_presence_false_when_only_feed_and_debris() -> None:
    w = DetectionWindow(max_frames=5)
    w.push(_feed_frame())
    w.push(_debris_frame())
    assert w.fish_presence is False


def test_fish_presence_stays_true_until_evicted() -> None:
    w = DetectionWindow(max_frames=2)
    w.push(_fish_frame())
    w.push(_empty_frame())
    assert w.fish_presence is True  # fish still in window

    w.push(_empty_frame())  # fish evicted
    assert w.fish_presence is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_invalid_max_frames_raises() -> None:
    with pytest.raises(ValueError):
        DetectionWindow(max_frames=0)


def test_max_frames_boundary() -> None:
    w = DetectionWindow(max_frames=1)
    w.push(_fish_frame())
    assert len(w) == 1
    w.push(_empty_frame())
    assert len(w) == 1
    assert w.fish_presence is False  # fish frame was evicted
