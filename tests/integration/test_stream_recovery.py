"""Integration tests for stream interruption and recovery behavior."""

import pytest
from datetime import datetime, timedelta
from core.stream_health import StreamHealthMonitor
from core.models import StreamHealthState


class TestStreamRecovery:
    """Test stream interruption and recovery scenarios."""

    def test_stream_interruption_detection(self):
        """Test detection of stream interruption."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()
        assert monitor.get_state() == StreamHealthState.ACTIVE

        monitor.last_frame_time = datetime.now() - timedelta(seconds=3.1)
        monitor.check_timeout(3)
        assert monitor.get_state() == StreamHealthState.INTERRUPTED

    def test_same_source_recovery(self):
        """Test recovery from interruption without application restart."""
        monitor = StreamHealthMonitor()

        monitor.update_frame_received()
        assert monitor.get_state() == StreamHealthState.ACTIVE

        monitor.last_frame_time = datetime.now() - timedelta(seconds=4)
        monitor.check_timeout(3)
        assert monitor.get_state() == StreamHealthState.INTERRUPTED

        monitor.update_frame_received()
        assert monitor.get_state() == StreamHealthState.ACTIVE

    def test_continuous_monitoring_during_interruption(self):
        """Test that monitoring continues during interruption."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()

        monitor.last_frame_time = datetime.now() - timedelta(seconds=4)
        monitor.check_timeout(3)
        assert monitor.get_state() == StreamHealthState.INTERRUPTED

        assert monitor.interrupted_at is not None

    def test_multiple_interruption_cycles(self):
        """Test multiple interruption and recovery cycles."""
        monitor = StreamHealthMonitor()

        for cycle in range(3):
            monitor.update_frame_received()
            assert monitor.get_state() == StreamHealthState.ACTIVE

            monitor.last_frame_time = datetime.now() - timedelta(seconds=4)
            monitor.check_timeout(3)
            assert monitor.get_state() == StreamHealthState.INTERRUPTED

    def test_no_false_interruption(self):
        """Test that frames within timeout do not trigger interruption."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()

        monitor.last_frame_time = datetime.now() - timedelta(seconds=2.5)
        monitor.check_timeout(3)
        assert monitor.get_state() == StreamHealthState.ACTIVE

    def test_recovery_without_restart(self):
        """Test that recovery does not require application restart."""
        monitor = StreamHealthMonitor()
        initial_session_id = id(monitor)

        monitor.update_frame_received()
        monitor.last_frame_time = datetime.now() - timedelta(seconds=4)
        monitor.check_timeout(3)

        monitor.update_frame_received()

        assert id(monitor) == initial_session_id
        assert monitor.get_state() == StreamHealthState.ACTIVE

    def test_interrupted_state_marker(self):
        """Test that interrupted time is properly marked."""
        monitor = StreamHealthMonitor()
        monitor.update_frame_received()

        monitor.last_frame_time = datetime.now() - timedelta(seconds=4)
        monitor.check_timeout(3)

        assert monitor.interrupted_at is not None

        monitor.update_frame_received()
        assert monitor.interrupted_at is None

    def test_transition_sequence(self):
        """Test proper state transition sequence during interruption."""
        monitor = StreamHealthMonitor()

        assert monitor.get_state() == StreamHealthState.WAITING

        monitor.update_frame_received()
        assert monitor.get_state() == StreamHealthState.ACTIVE

        monitor.last_frame_time = datetime.now() - timedelta(seconds=4)
        monitor.check_timeout(3)
        assert monitor.get_state() == StreamHealthState.INTERRUPTED

        monitor.update_frame_received()
        assert monitor.get_state() == StreamHealthState.ACTIVE

        monitor.stop()
        assert monitor.get_state() == StreamHealthState.WAITING
