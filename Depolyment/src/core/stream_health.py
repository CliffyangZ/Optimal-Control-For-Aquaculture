"""Stream and detection health status derivation."""

from datetime import datetime, timezone
from typing import Optional

from core.models import DetectionHealthStatus, StreamStatus


def _utc_now_matching(reference: datetime) -> datetime:
    if reference.tzinfo is None:
        return datetime.now()
    return datetime.now(timezone.utc)


def derive_stream_status(
    last_frame_at: Optional[datetime],
    started_at: Optional[datetime],
    interruption_timeout_seconds: int = 3,
) -> StreamStatus:
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
    detector_available: bool = True,
    loading: bool = False,
) -> DetectionHealthStatus:
    if not detector_available:
        return DetectionHealthStatus.UNAVAILABLE
    if loading or last_frame_at is None or last_overlay_at is None:
        return DetectionHealthStatus.LOADING
    age = (_utc_now_matching(last_overlay_at) - last_overlay_at).total_seconds()
    if age <= stale_seconds:
        return DetectionHealthStatus.ACTIVE
    return DetectionHealthStatus.STALE
