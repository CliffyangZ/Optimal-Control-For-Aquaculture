"""Stream health state derivation and transitions."""

from datetime import datetime
from typing import Optional
from core.models import StreamHealthState


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
