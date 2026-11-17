"""FeedingStateMachine: threshold-based 7-state controller for the BMduino feeder.

The machine is pure Python with no I/O — it receives `DetectionWindow` statistics
and emits `BMduinoCommand` values. BMduino serial communication is handled by an
external adapter layer (not implemented here).

Call `trigger_feeding()` to initiate a feeding cycle (IDLE → PRECHECK).
Call `update(window, last_detection_age_s)` on every detection loop tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.infer.window import DetectionWindow
from core.types import BMduinoCommand, FeedingState


@dataclass
class FeedingConfig:
    """Thresholds and timers for the FeedingStateMachine."""

    # CHECKING → DONE when mean_feed_count < this
    feed_threshold: float = 1.0
    # CHECKING → ALERT when mean_debris_count > this
    debris_threshold: float = 3.0
    # WAITING → CHECKING after this many seconds in WAITING
    wait_timer_s: float = 30.0
    # Any → ALERT when last_detection_age_s exceeds this
    camera_failure_timeout_s: float = 10.0
    # DISPENSE → WAITING after this many seconds (how long the feeder motor runs)
    dispense_duration_s: float = 2.0


class FeedingStateMachine:
    """7-state threshold-based feeding controller.

    States: IDLE, PRECHECK, DISPENSE, WAITING, CHECKING, DONE, ALERT.
    All 9 transitions from data-model.md §FeedingStateMachine are implemented.
    """

    def __init__(self, config: FeedingConfig) -> None:
        self.config = config
        self._state: FeedingState = FeedingState.IDLE
        self._state_entered_at: float = time.monotonic()

    @property
    def state(self) -> FeedingState:
        return self._state

    def trigger_feeding(self) -> BMduinoCommand | None:
        """External trigger: IDLE → PRECHECK (feeding schedule fired)."""
        if self._state == FeedingState.IDLE:
            self._transition(FeedingState.PRECHECK)
        return None

    def reset(self) -> None:
        """Reset to IDLE so a new feeding cycle can begin after DONE or ALERT."""
        self._transition(FeedingState.IDLE)

    def _transition(self, new_state: FeedingState) -> None:
        self._state = new_state
        self._state_entered_at = time.monotonic()

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_entered_at

    def update(
        self,
        window: DetectionWindow,
        last_detection_age_s: float,
    ) -> BMduinoCommand | None:
        """Evaluate window statistics and advance state; return a command if one is emitted.

        Camera failure watchdog fires from any non-terminal state when
        `last_detection_age_s` exceeds `config.camera_failure_timeout_s`.
        """
        # Camera failure watchdog — applies from any non-terminal state
        terminal = {FeedingState.DONE, FeedingState.ALERT}
        if (
            self._state not in terminal
            and last_detection_age_s > self.config.camera_failure_timeout_s
        ):
            self._transition(FeedingState.ALERT)
            return BMduinoCommand.ALERT

        state = self._state

        if state == FeedingState.IDLE:
            return None

        if state == FeedingState.PRECHECK:
            if window.mean_feed_count < self.config.feed_threshold:
                # No residual feed — safe to dispense
                self._transition(FeedingState.DISPENSE)
                return BMduinoCommand.FEED
            else:
                # Previous dose not consumed — skip dispense, wait
                self._transition(FeedingState.WAITING)
                return None

        if state == FeedingState.DISPENSE:
            # Hold DISPENSE until the motor has run for dispense_duration_s, then stop.
            if self._time_in_state() >= self.config.dispense_duration_s:
                self._transition(FeedingState.WAITING)
                return BMduinoCommand.STOP
            return None

        if state == FeedingState.WAITING:
            if self._time_in_state() >= self.config.wait_timer_s:
                self._transition(FeedingState.CHECKING)
            return None

        if state == FeedingState.CHECKING:
            if window.mean_debris_count > self.config.debris_threshold:
                self._transition(FeedingState.ALERT)
                return BMduinoCommand.ALERT
            if window.mean_feed_count < self.config.feed_threshold:
                self._transition(FeedingState.DONE)
                return None
            # Feed still present — go back to waiting
            self._transition(FeedingState.WAITING)
            return None

        # DONE and ALERT are terminal — no further transitions
        return None
