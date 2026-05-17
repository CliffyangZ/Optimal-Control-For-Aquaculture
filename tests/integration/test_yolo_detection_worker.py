"""Integration tests: YOLO detection worker broadcasts detections.v1 WebSocket messages.

Mocks `YOLODetector.detect()` to return a synthetic 2-detection `DetectionResult`
so no GPU or model file is required.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.models import DetectionHealthStatus, StreamStatus
from core.types import Detection, DetectionResult
from web.detection_worker import DetectionWorker
from web.overlay_hub import overlay_hub


def _make_detection_result() -> DetectionResult:
    return DetectionResult(
        frame_id=42,
        captured_at=datetime.now(timezone.utc),
        source_width=1280,
        source_height=720,
        detections=[
            Detection(
                class_name="fish",
                class_index=2,
                confidence=0.921,
                bbox_x=0.312,
                bbox_y=0.451,
                bbox_w=0.087,
                bbox_h=0.094,
            ),
            Detection(
                class_name="feed",
                class_index=1,
                confidence=0.873,
                bbox_x=0.654,
                bbox_y=0.213,
                bbox_w=0.043,
                bbox_h=0.039,
            ),
        ],
        inference_ms=15.0,
    )


# ---------------------------------------------------------------------------
# Broadcast contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_contains_detections_v1_schema() -> None:
    """One detection loop iteration must produce a detections.v1 WebSocket message."""
    worker = DetectionWorker()

    mock_detector = MagicMock()
    mock_detector.detect.return_value = _make_detection_result()
    worker._detector = mock_detector

    broadcast_calls: list[dict] = []
    original_broadcast = overlay_hub.broadcast

    def capture_broadcast(msg: dict) -> None:
        broadcast_calls.append(msg)

    with patch.object(overlay_hub, "broadcast", side_effect=capture_broadcast):
        session = await worker.register_viewer("viewer-1")
        assert session is not None

        from core.models import VideoFrame
        from streaming.latest_frame_reader import LatestFrameReader
        import numpy as np

        # Build a minimal fake raw_frame matching what LatestFrameReader returns
        fake_frame = MagicMock()
        fake_frame.sequence_number = 42
        fake_frame.captured_at = datetime.now(timezone.utc)
        fake_frame.source_width = 1280
        fake_frame.source_height = 720
        fake_frame.image = np.zeros((720, 1280, 3), dtype=np.uint8)

        worker._frame_reader = MagicMock(spec=LatestFrameReader)
        worker._frame_reader.get_latest_frame.return_value = fake_frame
        worker._frame_reader.get_last_frame_time.return_value = datetime.now(timezone.utc)

        # Drive one iteration of the detection loop
        import time
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        video_frame = VideoFrame(
            frame_id=fake_frame.sequence_number,
            captured_at=fake_frame.captured_at,
            source_width=fake_frame.source_width,
            source_height=fake_frame.source_height,
            image=fake_frame.image,
        )
        result = _make_detection_result()
        worker._session = session
        worker._session.detection_status = DetectionHealthStatus.ACTIVE
        await worker._broadcast_overlay(video_frame, result, now_ms)

    # Assert a detections.v1 message was broadcast
    assert len(broadcast_calls) == 1
    msg = broadcast_calls[0]
    assert msg["schema"] == "detections.v1"


@pytest.mark.asyncio
async def test_broadcast_detections_list_has_correct_fields() -> None:
    worker = DetectionWorker()
    session = await worker.register_viewer("viewer-2")

    from core.models import VideoFrame
    import numpy as np
    video_frame = VideoFrame(
        frame_id=99,
        captured_at=datetime.now(timezone.utc),
        source_width=640,
        source_height=480,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    result = _make_detection_result()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    broadcast_calls: list[dict] = []
    with patch.object(overlay_hub, "broadcast", side_effect=lambda m: broadcast_calls.append(m)):
        worker._session = session
        worker._session.detection_status = DetectionHealthStatus.ACTIVE
        await worker._broadcast_overlay(video_frame, result, now_ms)

    assert len(broadcast_calls) == 1
    msg = broadcast_calls[0]
    dets = msg["detections"]
    assert len(dets) == 2

    fish = next(d for d in dets if d["class_name"] == "fish")
    feed = next(d for d in dets if d["class_name"] == "feed")

    assert fish["confidence"] == pytest.approx(0.921, abs=0.001)
    assert feed["class_name"] == "feed"

    for d in dets:
        for field in ("class_name", "confidence", "x", "y", "width", "height"):
            assert field in d, f"Missing field '{field}' in detection"


@pytest.mark.asyncio
async def test_broadcast_empty_detections_is_valid() -> None:
    worker = DetectionWorker()
    session = await worker.register_viewer("viewer-3")

    from core.models import VideoFrame
    import numpy as np

    video_frame = VideoFrame(
        frame_id=1,
        captured_at=datetime.now(timezone.utc),
        source_width=640,
        source_height=480,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    empty_result = DetectionResult(
        frame_id=1,
        captured_at=datetime.now(timezone.utc),
        source_width=640,
        source_height=480,
        detections=[],
        inference_ms=10.0,
    )

    broadcast_calls: list[dict] = []
    with patch.object(overlay_hub, "broadcast", side_effect=lambda m: broadcast_calls.append(m)):
        worker._session = session
        worker._session.detection_status = DetectionHealthStatus.ACTIVE
        await worker._broadcast_overlay(video_frame, empty_result, 12345)

    assert len(broadcast_calls) == 1
    assert broadcast_calls[0]["detections"] == []
    assert broadcast_calls[0]["schema"] == "detections.v1"
