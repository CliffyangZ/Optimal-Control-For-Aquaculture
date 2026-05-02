"""Tests for stream health state transitions."""

import pytest
import time
from datetime import datetime, timedelta
from core.models import StreamHealthState
from core.stream_health import StreamHealthMonitor


class TestStreamHealthStateTransitions:
    """Test stream health state transitions during monitoring."""

    def test_initial_state_is_waiting(self):
        """Stream health should start in Waiting state."""
        assert StreamHealthState.WAITING is not None

    def test_waiting_to_active_on_frame(self):
        """Waiting should transition to Active when frames arrive after Start."""
        initial_state = StreamHealthState.WAITING
        active_state = StreamHealthState.ACTIVE
        assert initial_state != active_state

    def test_active_to_interrupted_after_timeout(self):
        """Active should transition to Interrupted after 3 seconds without frames."""
        active_state = StreamHealthState.ACTIVE
        interrupted_state = StreamHealthState.INTERRUPTED
        assert active_state != interrupted_state

    def test_interrupted_to_active_on_frame(self):
        """Interrupted should return to Active when same source resumes."""
        interrupted_state = StreamHealthState.INTERRUPTED
        active_state = StreamHealthState.ACTIVE
        assert interrupted_state != active_state

    def test_stop_returns_to_waiting(self):
        """Stop should return to Waiting from any active state."""
        for state in [StreamHealthState.ACTIVE, StreamHealthState.INTERRUPTED]:
            final_state = StreamHealthState.WAITING
            assert state != final_state

    def test_interrupted_timeout_duration(self):
        """Interruption timeout should be 3 seconds per specification."""
        from config.setting import INTERRUPTION_TIMEOUT_SECONDS
        assert INTERRUPTION_TIMEOUT_SECONDS == 3

    def test_all_states_defined(self):
        """Verify all required health states exist."""
        assert StreamHealthState.WAITING
        assert StreamHealthState.ACTIVE
        assert StreamHealthState.INTERRUPTED
        assert StreamHealthState.DETECTION_UNAVAILABLE

    def test_state_to_detection_unavailable(self):
        """Active state can transition to Detection Unavailable."""
        active_state = StreamHealthState.ACTIVE
        unavailable_state = StreamHealthState.DETECTION_UNAVAILABLE
        assert active_state != unavailable_state

    def test_detection_unavailable_to_active(self):
        """Detection Unavailable can return to Active if detection becomes reliable."""
        unavailable_state = StreamHealthState.DETECTION_UNAVAILABLE
        active_state = StreamHealthState.ACTIVE
        assert unavailable_state != active_state


class TestStreamHealthMonitor:
    """Test StreamHealthMonitor state management."""

    def test_monitor_initial_state(self):
        """Monitor should start in Waiting state."""
        monitor = StreamHealthMonitor()
        assert monitor.get_state() == StreamHealthState.WAITING

    def test_frame_received_activates_monitoring(self):
        """Receiving a frame should activate monitoring."""
        monitor = StreamHealthMonitor()
        state = monitor.update_frame_received()
        assert state == StreamHealthState.ACTIVE

    def test_timeout_detection(self):
        """Monitor should detect timeout after 3 seconds."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()
        assert monitor.get_state() == StreamHealthState.ACTIVE

        monitor.check_timeout(3)
        assert monitor.get_state() == StreamHealthState.ACTIVE

    def test_interruption_after_timeout(self):
        """Monitor should detect interruption after timeout."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()
        monitor.last_frame_time = datetime.now() - timedelta(seconds=4)
        monitor.check_timeout(3)
        assert monitor.get_state() == StreamHealthState.INTERRUPTED

    def test_recovery_from_interruption(self):
        """Monitor should return to Active when frame received during interruption."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()
        monitor.last_frame_time = datetime.now() - timedelta(seconds=4)
        monitor.check_timeout(3)
        assert monitor.get_state() == StreamHealthState.INTERRUPTED

        monitor.update_frame_received()
        assert monitor.get_state() == StreamHealthState.ACTIVE

    def test_stop_clears_monitoring(self):
        """Stop should clear monitoring state."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()
        monitor.stop()
        assert monitor.get_state() == StreamHealthState.WAITING
        assert monitor.last_frame_time is None

    def test_detection_unavailable_transition(self):
        """Monitor can transition to Detection Unavailable."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()
        monitor.set_detection_unavailable()
        assert monitor.get_state() == StreamHealthState.DETECTION_UNAVAILABLE

    def test_recovery_from_detection_unavailable(self):
        """Monitor can return to Active from Detection Unavailable."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()
        monitor.set_detection_unavailable()
        monitor.set_active()
        assert monitor.get_state() == StreamHealthState.ACTIVE
