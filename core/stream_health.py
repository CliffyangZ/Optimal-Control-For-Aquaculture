"""Stream / detection health derivation helpers.

These pure functions derive the user-facing `StreamStatus` and
`DetectionHealthStatus` from raw frame / overlay timing. They are used by the
detection worker to keep `LiveStreamSession.stream_status` and
`LiveStreamSession.detection_status` in sync with reality before each overlay
broadcast.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.models import DetectionHealthStatus, StreamStatus


def _utc_now_matching(reference: datetime) -> datetime:
    """Return a `now()` with the same tzinfo awareness as `reference`."""
    if reference.tzinfo is None:
        return datetime.now()
    return datetime.now(timezone.utc)


def derive_stream_status(
    last_frame_at: Optional[datetime],
    started_at: Optional[datetime],
    interruption_timeout_seconds: int = 3,
) -> StreamStatus:
    """Derive `StreamStatus` from frame-timing information.

    - No `started_at`              → WAITING
    - Started but no frames yet    → CONNECTING (within timeout) / STREAM_UNAVAILABLE (after)
    - Recent frames                → LIVE
    - Stale frames                 → RECONNECTING
    """
    if started_at is None:
        return StreamStatus.WAITING
    now = _utc_now_matching(started_at)
    if last_frame_at is None:
        elapsed = (now - started_at).total_seconds()
        if elapsed < interruption_timeout_seconds:
            return StreamStatus.CONNECTING
        return StreamStatus.STREAM_UNAVAILABLE
    age = (_utc_now_matching(last_frame_at) - last_frame_at).total_seconds()
    if age <= interruption_timeout_seconds:
        return StreamStatus.LIVE
    return StreamStatus.RECONNECTING


def derive_detection_status(
    last_overlay_at: Optional[datetime],
    last_frame_at: Optional[datetime],
    stale_seconds: float = 2.0,
) -> DetectionHealthStatus:
    """Derive `DetectionHealthStatus` from overlay / frame timing.

    - No frames or overlays yet      → UNAVAILABLE
    - Recent overlay                 → OK
    - Slightly stale (≤ 5× stale)    → DELAYED
    - Very stale (> 5× stale)        → DEGRADED
    """
    if last_frame_at is None or last_overlay_at is None:
        return DetectionHealthStatus.UNAVAILABLE
    age = (_utc_now_matching(last_overlay_at) - last_overlay_at).total_seconds()
    if age <= stale_seconds:
        return DetectionHealthStatus.OK
    if age <= stale_seconds * 5:
        return DetectionHealthStatus.DELAYED
    return DetectionHealthStatus.DEGRADED
