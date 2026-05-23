"""Unit tests for `core.infer.state_machine.FeedingStateMachine`.

Covers:
- IDLE → PRECHECK → DISPENSE → WAITING → CHECKING → DONE full path
- CHECKING → ALERT on high debris
- Camera failure ALERT trigger
- PRECHECK → WAITING when feed detected before dispense
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from core.infer.state_machine import FeedingConfig, FeedingStateMachine
from core.infer.window import DetectionWindow
from core.types import BMduinoCommand, Detection, DetectionResult, FeedingState


def _make_config(**kwargs) -> FeedingConfig:
    defaults = dict(
        feed_threshold=1.0,
        debris_threshold=3.0,
        wait_timer_s=30.0,
        camera_failure_timeout_s=10.0,
        dispense_duration_s=2.0,
    )
    defaults.update(kwargs)
    return FeedingConfig(**defaults)


def _empty_window() -> DetectionWindow:
    w = DetectionWindow(max_frames=5)
    return w


def _window_with_feed(count: int = 2) -> DetectionWindow:
    w = DetectionWindow(max_frames=5)
    result = _make_result([("feed", 1)] * count)
    for _ in range(5):
        w.push(result)
    return w


def _window_with_debris(count: int = 4) -> DetectionWindow:
    w = DetectionWindow(max_frames=5)
    result = _make_result([("debris", 0)] * count)
    for _ in range(5):
        w.push(result)
    return w


def _make_result(detections: list[tuple[str, int]]) -> DetectionResult:
    dets = [
        Detection(
            class_name=name,
            class_index=idx,
            confidence=0.9,
            bbox_x=0.1,
            bbox_y=0.1,
            bbox_w=0.1,
            bbox_h=0.1,
        )
        for name, idx in detections
    ]
    return DetectionResult(
        frame_id=0,
        captured_at=datetime.now(timezone.utc),
        source_width=640,
        source_height=480,
        detections=dets,
        inference_ms=5.0,
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_is_idle() -> None:
    fsm = FeedingStateMachine(_make_config())
    assert fsm.state == FeedingState.IDLE


def test_update_in_idle_returns_none() -> None:
    fsm = FeedingStateMachine(_make_config())
    result = fsm.update(_empty_window(), last_detection_age_s=0.1)
    assert result is None
    assert fsm.state == FeedingState.IDLE


# ---------------------------------------------------------------------------
# Full happy path: IDLE → PRECHECK → DISPENSE → WAITING → CHECKING → DONE
# ---------------------------------------------------------------------------

def test_trigger_transitions_idle_to_precheck() -> None:
    fsm = FeedingStateMachine(_make_config())
    fsm.trigger_feeding()
    assert fsm.state == FeedingState.PRECHECK


def test_precheck_to_dispense_when_no_feed() -> None:
    fsm = FeedingStateMachine(_make_config())
    fsm.trigger_feeding()
    cmd = fsm.update(_empty_window(), last_detection_age_s=0.0)
    assert fsm.state == FeedingState.DISPENSE
    assert cmd == BMduinoCommand.FEED


def test_dispense_holds_until_timer_expires() -> None:
    """DISPENSE must stay active for dispense_duration_s before sending STOP."""
    with patch("core.infer.state_machine.time") as mock_time:
        mock_time.monotonic.side_effect = [
            0.0,  # __init__
            1.0,  # _transition(PRECHECK) in trigger_feeding
            2.0,  # _transition(DISPENSE) in PRECHECK update
            2.1,  # _time_in_state(): 0.1s < 2.0s → stay in DISPENSE
            4.5,  # _time_in_state(): 2.5s >= 2.0s → transition
            5.0,  # _transition(WAITING)
        ]
        fsm = FeedingStateMachine(_make_config(dispense_duration_s=2.0))
        fsm.trigger_feeding()

        cmd1 = fsm.update(_empty_window(), last_detection_age_s=0.0)  # PRECHECK → DISPENSE
        assert fsm.state == FeedingState.DISPENSE
        assert cmd1 == BMduinoCommand.FEED

        cmd2 = fsm.update(_empty_window(), last_detection_age_s=0.0)  # DISPENSE (not expired)
        assert fsm.state == FeedingState.DISPENSE
        assert cmd2 is None

        cmd3 = fsm.update(_empty_window(), last_detection_age_s=0.0)  # DISPENSE → WAITING
        assert fsm.state == FeedingState.WAITING
        assert cmd3 == BMduinoCommand.STOP


def test_waiting_to_checking_after_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate timer expiry by manually setting state and entered_at, then patching time
    with patch("core.infer.state_machine.time") as mock_time:
        # __init__ calls time.monotonic() once, then _time_in_state() + _transition each once
        mock_time.monotonic.side_effect = [
            0.0,   # __init__: _state_entered_at (overridden below)
            6.0,   # _time_in_state(): 6.0 - 0.0 = 6.0 >= wait_timer_s(5.0) → transition
            7.0,   # _transition(CHECKING) records new _state_entered_at
        ]
        fsm2 = FeedingStateMachine(_make_config(wait_timer_s=5.0))
        fsm2._state = FeedingState.WAITING
        fsm2._state_entered_at = 0.0  # entered WAITING at t=0
        result = fsm2.update(_empty_window(), last_detection_age_s=0.0)
    assert fsm2.state == FeedingState.CHECKING
    assert result is None


def test_checking_to_done_when_feed_gone() -> None:
    fsm = FeedingStateMachine(_make_config())
    fsm._state = FeedingState.CHECKING
    fsm._state_entered_at = 0.0

    result = fsm.update(_empty_window(), last_detection_age_s=0.0)
    assert fsm.state == FeedingState.DONE
    assert result is None


# ---------------------------------------------------------------------------
# PRECHECK → WAITING when feed detected (previous dose not consumed)
# ---------------------------------------------------------------------------

def test_precheck_to_waiting_when_feed_detected() -> None:
    fsm = FeedingStateMachine(_make_config(feed_threshold=1.0))
    fsm.trigger_feeding()
    # Window has feed present (mean_feed_count = 2.0 > threshold 1.0)
    cmd = fsm.update(_window_with_feed(2), last_detection_age_s=0.0)
    assert fsm.state == FeedingState.WAITING
    assert cmd is None


# ---------------------------------------------------------------------------
# CHECKING → ALERT on high debris
# ---------------------------------------------------------------------------

def test_checking_to_alert_on_high_debris() -> None:
    fsm = FeedingStateMachine(_make_config(debris_threshold=3.0))
    fsm._state = FeedingState.CHECKING
    fsm._state_entered_at = 0.0

    # Window has 4 debris per frame (> threshold 3.0)
    cmd = fsm.update(_window_with_debris(4), last_detection_age_s=0.0)
    assert fsm.state == FeedingState.ALERT
    assert cmd == BMduinoCommand.ALERT


# ---------------------------------------------------------------------------
# CHECKING → WAITING when feed still present
# ---------------------------------------------------------------------------

def test_checking_to_waiting_when_feed_remains() -> None:
    fsm = FeedingStateMachine(_make_config(feed_threshold=1.0, debris_threshold=10.0))
    fsm._state = FeedingState.CHECKING
    fsm._state_entered_at = 0.0

    cmd = fsm.update(_window_with_feed(2), last_detection_age_s=0.0)
    assert fsm.state == FeedingState.WAITING
    assert cmd is None


# ---------------------------------------------------------------------------
# Camera failure ALERT trigger
# ---------------------------------------------------------------------------

def test_camera_failure_triggers_alert_from_precheck() -> None:
    fsm = FeedingStateMachine(_make_config(camera_failure_timeout_s=5.0))
    fsm.trigger_feeding()
    assert fsm.state == FeedingState.PRECHECK

    cmd = fsm.update(_empty_window(), last_detection_age_s=6.0)
    assert fsm.state == FeedingState.ALERT
    assert cmd == BMduinoCommand.ALERT


def test_camera_failure_triggers_alert_from_waiting() -> None:
    fsm = FeedingStateMachine(_make_config(camera_failure_timeout_s=5.0))
    fsm._state = FeedingState.WAITING

    cmd = fsm.update(_empty_window(), last_detection_age_s=6.0)
    assert fsm.state == FeedingState.ALERT
    assert cmd == BMduinoCommand.ALERT


def test_camera_failure_triggers_alert_from_checking() -> None:
    fsm = FeedingStateMachine(_make_config(camera_failure_timeout_s=5.0))
    fsm._state = FeedingState.CHECKING

    cmd = fsm.update(_empty_window(), last_detection_age_s=6.0)
    assert fsm.state == FeedingState.ALERT
    assert cmd == BMduinoCommand.ALERT


def test_camera_failure_does_not_trigger_from_done() -> None:
    """DONE is terminal; camera failure watchdog should not apply."""
    fsm = FeedingStateMachine(_make_config(camera_failure_timeout_s=5.0))
    fsm._state = FeedingState.DONE

    cmd = fsm.update(_empty_window(), last_detection_age_s=100.0)
    assert fsm.state == FeedingState.DONE
    assert cmd is None


def test_camera_failure_does_not_trigger_from_alert() -> None:
    """ALERT is terminal; re-entering ALERT should be idempotent."""
    fsm = FeedingStateMachine(_make_config(camera_failure_timeout_s=5.0))
    fsm._state = FeedingState.ALERT

    cmd = fsm.update(_empty_window(), last_detection_age_s=100.0)
    assert fsm.state == FeedingState.ALERT
    assert cmd is None


# ---------------------------------------------------------------------------
# trigger_feeding guard
# ---------------------------------------------------------------------------

def test_trigger_feeding_only_works_from_idle() -> None:
    """trigger_feeding should be a no-op when not in IDLE."""
    fsm = FeedingStateMachine(_make_config())
    fsm._state = FeedingState.CHECKING
    fsm.trigger_feeding()
    assert fsm.state == FeedingState.CHECKING


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_from_done_returns_to_idle() -> None:
    fsm = FeedingStateMachine(_make_config())
    fsm._state = FeedingState.DONE
    fsm.reset()
    assert fsm.state == FeedingState.IDLE


def test_reset_from_alert_returns_to_idle() -> None:
    fsm = FeedingStateMachine(_make_config())
    fsm._state = FeedingState.ALERT
    fsm.reset()
    assert fsm.state == FeedingState.IDLE


def test_trigger_feeding_works_after_reset() -> None:
    fsm = FeedingStateMachine(_make_config())
    fsm._state = FeedingState.DONE
    fsm.reset()
    fsm.trigger_feeding()
    assert fsm.state == FeedingState.PRECHECK
