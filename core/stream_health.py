"""Stream health state derivation and transitions."""

from datetime import datetime, timezone
from typing import Optional
from core.models import StreamHealthState, StreamStatus, DetectionHealthStatus


class StreamHealthMonitor:
    """Monitors and derives stream health state."""

    def __init__(self):
        self.last_frame_time: Optional[datetime] = None
        self.current_state = StreamHealthState.WAITING
        self.interrupted_at: Optional[datetime] = None

    def update_frame_received(self) -> StreamHealthState:
        """Update when a frame is received."""
        self.last_frame_time = datetime.now()
        if self.current_state == StreamHealthState.INTERRUPTED:
            self.current_state = StreamHealthState.ACTIVE
            self.interrupted_at = None
        elif self.current_state != StreamHealthState.ACTIVE:
            self.current_state = StreamHealthState.ACTIVE
        return self.current_state

    def check_timeout(self, timeout_seconds: int) -> StreamHealthState:
        """Check if stream has timed out."""
        if self.last_frame_time is None:
            return self.current_state

        time_since_frame = (datetime.now() - self.last_frame_time).total_seconds()
        if time_since_frame > timeout_seconds:
            if self.current_state == StreamHealthState.ACTIVE:
                self.current_state = StreamHealthState.INTERRUPTED
                self.interrupted_at = datetime.now()

        return self.current_state

    def set_detection_unavailable(self) -> StreamHealthState:
        """Set state to Detection Unavailable while frames are current."""
        if self.current_state == StreamHealthState.ACTIVE:
            self.current_state = StreamHealthState.DETECTION_UNAVAILABLE
        return self.current_state

    def set_active(self) -> StreamHealthState:
        """Return to Active state if detection becomes available."""
        if self.current_state == StreamHealthState.DETECTION_UNAVAILABLE:
            self.current_state = StreamHealthState.ACTIVE
        return self.current_state

    def stop(self) -> StreamHealthState:
        """Stop monitoring and return to Waiting."""
        self.current_state = StreamHealthState.WAITING
        self.last_frame_time = None
        self.interrupted_at = None
        return self.current_state

    def get_state(self) -> StreamHealthState:
        """Get current health state."""
        return self.current_state


def _utc_now_matching(reference: datetime) -> datetime:
    """Return a now() with the same tzinfo awareness as reference."""
    if reference.tzinfo is None:
        return datetime.now()
    return datetime.now(timezone.utc)


def derive_stream_status(
    last_frame_at: Optional[datetime],
    started_at: Optional[datetime],
    interruption_timeout_seconds: int = 3,
) -> StreamStatus:
    """Derive StreamStatus from frame timing."""
    if started_at is None:
        return StreamStatus.WAITING
    now = _utc_now_matching(started_at)
    if last_frame_at is None:
        elapsed = (now - started_at).total_seconds()
        return StreamStatus.CONNECTING if elapsed < interruption_timeout_seconds else StreamStatus.STREAM_UNAVAILABLE
    age = (_utc_now_matching(last_frame_at) - last_frame_at).total_seconds()
    if age <= interruption_timeout_seconds:
        return StreamStatus.LIVE
    return StreamStatus.RECONNECTING


def derive_detection_status(
    last_overlay_at: Optional[datetime],
    last_frame_at: Optional[datetime],
    stale_seconds: float = 2.0,
) -> DetectionHealthStatus:
    """Derive DetectionHealthStatus from overlay/frame timing."""
    if last_frame_at is None:
        return DetectionHealthStatus.UNAVAILABLE
    if last_overlay_at is None:
        return DetectionHealthStatus.UNAVAILABLE
    age = (_utc_now_matching(last_overlay_at) - last_overlay_at).total_seconds()
    if age <= stale_seconds:
        return DetectionHealthStatus.OK
    if age <= stale_seconds * 5:
        return DetectionHealthStatus.DELAYED
    return DetectionHealthStatus.DEGRADED
