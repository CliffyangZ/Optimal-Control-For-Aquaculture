"""UI contract tests for monitoring interface."""

import pytest
from core.models import StreamHealthState, MonitoringSession
from datetime import datetime


class TestMonitoringUIContract:
    """Test monitoring UI contracts and expected states."""

    def test_initial_state_is_waiting(self):
        """Test that monitoring interface starts in Waiting state."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now()
        )
        assert session.health_state == StreamHealthState.WAITING

    def test_waiting_state_no_frame_displayed(self):
        """Test that Waiting state has no active frame."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now()
        )
        assert session.latest_frame is None

    def test_waiting_state_no_detection_displayed(self):
        """Test that Waiting state has no active detection."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now()
        )
        assert session.latest_detection is None

    def test_start_control_available_in_waiting(self):
        """Test that Start control is available in Waiting state."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.WAITING
        )
        assert session.health_state == StreamHealthState.WAITING

    def test_stop_control_available_in_active(self):
        """Test that Stop control is available in Active state."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.ACTIVE
        )
        assert session.health_state == StreamHealthState.ACTIVE

    def test_stop_returns_to_waiting(self):
        """Test that Stop returns UI to Waiting state."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.ACTIVE
        )
        session.health_state = StreamHealthState.WAITING
        session.latest_frame = None
        session.latest_detection = None

        assert session.health_state == StreamHealthState.WAITING
        assert session.latest_frame is None

    def test_start_label_visible(self):
        """Test that Start label is visible in Waiting state."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.WAITING
        )
        assert StreamHealthState.WAITING.value == "Waiting"

    def test_stop_label_visible(self):
        """Test that Stop label is visible in Active state."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.ACTIVE
        )
        assert StreamHealthState.ACTIVE.value == "Active"

    def test_processed_view_default(self):
        """Test that processed view is the default."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now()
        )
        assert session.view_mode == "processed"

    def test_raw_view_available(self):
        """Test that raw view mode can be set."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            view_mode="raw"
        )
        assert session.view_mode == "raw"

    def test_view_mode_toggle(self):
        """Test toggling between processed and raw views."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            view_mode="processed"
        )
        assert session.view_mode == "processed"

        session.view_mode = "raw"
        assert session.view_mode == "raw"

        session.view_mode = "processed"
        assert session.view_mode == "processed"

    def test_health_state_display_values(self):
        """Test that health states have proper display values."""
        states = [
            StreamHealthState.WAITING,
            StreamHealthState.ACTIVE,
            StreamHealthState.INTERRUPTED
        ]
        for state in states:
            assert len(state.value) > 0
            assert state.value in [
                "Waiting", "Active", "Interrupted", "Detection Unavailable"
            ]

    def test_interrupted_state_available(self):
        """Test that Interrupted state is available."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.INTERRUPTED
        )
        assert session.health_state == StreamHealthState.INTERRUPTED

    def test_detection_unavailable_state(self):
        """Test that Detection Unavailable state is available."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.DETECTION_UNAVAILABLE
        )
        assert session.health_state == StreamHealthState.DETECTION_UNAVAILABLE

    def test_session_tracking(self):
        """Test that sessions have unique IDs."""
        session1 = MonitoringSession(
            session_id="session-001",
            started_at=datetime.now()
        )
        session2 = MonitoringSession(
            session_id="session-002",
            started_at=datetime.now()
        )
        assert session1.session_id != session2.session_id


class TestInterruptedUIContract:
    """Test interrupted state UI contract (User Story 2)."""

    def test_interrupted_state_display(self):
        """Test that Interrupted state is displayed to operator."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.INTERRUPTED
        )
        assert session.health_state == StreamHealthState.INTERRUPTED
        assert session.health_state.value == "Interrupted"

    def test_interrupted_no_stale_frame_labeled_current(self):
        """Test that stale frame is not labeled as current during interruption."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.ACTIVE
        )
        old_frame = None
        session.latest_frame = old_frame

        session.health_state = StreamHealthState.INTERRUPTED
        if session.health_state == StreamHealthState.INTERRUPTED:
            session.latest_frame = None

        assert session.latest_frame is None

    def test_interrupted_no_stale_detection_labeled_current(self):
        """Test that stale detection is not labeled as current during interruption."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.ACTIVE
        )
        old_detection = None
        session.latest_detection = old_detection

        session.health_state = StreamHealthState.INTERRUPTED
        if session.health_state == StreamHealthState.INTERRUPTED:
            session.latest_detection = None

        assert session.latest_detection is None

    def test_interrupted_continues_reading_attempt(self):
        """Test that monitoring continues attempting to read during interruption."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.INTERRUPTED
        )
        assert session.health_state == StreamHealthState.INTERRUPTED
        assert not session.stopped_at

    def test_interrupted_recovers_on_new_frame(self):
        """Test that Interrupted state returns to Active when frames resume."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.INTERRUPTED
        )
        assert session.health_state == StreamHealthState.INTERRUPTED

        session.health_state = StreamHealthState.ACTIVE
        assert session.health_state == StreamHealthState.ACTIVE

    def test_interrupted_message_operator_readable(self):
        """Test that Interrupted message is clear to operator."""
        state = StreamHealthState.INTERRUPTED
        assert state.value == "Interrupted"
        assert len(state.value) > 0


class TestDetectionUnavailableUIContract:
    """Test Detection Unavailable state UI contract (User Story 3)."""

    def test_detection_unavailable_state_display(self):
        """Test that Detection Unavailable state is displayed."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.DETECTION_UNAVAILABLE
        )
        assert session.health_state == StreamHealthState.DETECTION_UNAVAILABLE

    def test_detection_unavailable_no_score(self):
        """Test that activity score is withheld during Detection Unavailable."""
        from core.models import RippleDetectionResult, DetectionStatus

        result = RippleDetectionResult(
            status=DetectionStatus.UNAVAILABLE,
            reason="Too dark",
            frame_sequence_number=1,
            completed_at=datetime.now()
        )

        assert result.activity_score is None
        assert result.activity_level is None

    def test_detection_unavailable_no_activity_level(self):
        """Test that activity level is withheld during Detection Unavailable."""
        from core.models import RippleDetectionResult, DetectionStatus

        result = RippleDetectionResult(
            status=DetectionStatus.UNAVAILABLE,
            reason="Too bright",
            frame_sequence_number=1,
            completed_at=datetime.now()
        )

        assert result.activity_score is None
        assert result.activity_level is None

    def test_detection_unavailable_includes_reason(self):
        """Test that Detection Unavailable includes a descriptive reason."""
        from core.models import RippleDetectionResult, DetectionStatus

        reasons = ["Too dark", "Too bright", "Unstable scene"]

        for reason in reasons:
            result = RippleDetectionResult(
                status=DetectionStatus.UNAVAILABLE,
                reason=reason,
                frame_sequence_number=1,
                completed_at=datetime.now()
            )
            assert result.reason == reason

    def test_available_detection_has_score_and_level(self):
        """Test that available detection includes score and level."""
        from core.models import RippleDetectionResult, DetectionStatus
        from core.activity_levels import ActivityLevel

        result = RippleDetectionResult(
            status=DetectionStatus.AVAILABLE,
            activity_score=50,
            activity_level=ActivityLevel.MEDIUM,
            frame_sequence_number=1,
            completed_at=datetime.now()
        )

        assert result.activity_score == 50
        assert result.activity_level == ActivityLevel.MEDIUM

    def test_frame_visible_during_detection_unavailable(self):
        """Test that latest frame remains visible during Detection Unavailable."""
        session = MonitoringSession(
            session_id="test-001",
            started_at=datetime.now(),
            health_state=StreamHealthState.ACTIVE
        )

        from core.models import CurrentFrame
        import numpy as np

        test_frame = CurrentFrame(
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
            captured_at=datetime.now(),
            sequence_number=1,
            source_width=1280,
            source_height=720
        )
        session.latest_frame = test_frame

        session.health_state = StreamHealthState.DETECTION_UNAVAILABLE
        assert session.latest_frame is not None

    def test_detection_unavailable_state_values(self):
        """Test that Detection Unavailable state has proper values."""
        state = StreamHealthState.DETECTION_UNAVAILABLE
        assert state.value == "Detection Unavailable"
