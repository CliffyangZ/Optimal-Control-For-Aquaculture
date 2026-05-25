"""Integration tests: RF-DETR detection worker broadcasts detections.v1 WebSocket messages.

Mocks `RFDETRDetector.detect()` to return a synthetic 2-detection `DetectionResult`
so no GPU or model file is required.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.models import DetectionHealthStatus, StreamStatus
from core.models import LiveStreamSession
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

    def capture_broadcast(msg: dict) -> None:
        broadcast_calls.append(msg)

    with patch.object(worker, "_start_worker"), patch.object(overlay_hub, "broadcast", side_effect=capture_broadcast):
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
    with patch.object(worker, "_start_worker"):
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
    assert fish["class_index"] == 2
    assert feed["class_name"] == "feed"

    for d in dets:
        for field in ("class_name", "class_index", "confidence", "x", "y", "width", "height"):
            assert field in d, f"Missing field '{field}' in detection"


@pytest.mark.asyncio
async def test_broadcast_empty_detections_is_valid() -> None:
    worker = DetectionWorker()
    with patch.object(worker, "_start_worker"):
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


def test_missing_rfdetr_model_path_sets_unavailable_without_fallback(monkeypatch) -> None:
    worker = DetectionWorker()
    worker._session = MagicMock()
    monkeypatch.setattr("config.setting.RFDETR_MODEL_PATH", "")

    worker._load_detector()

    assert worker._detector is None
    assert worker._session.detection_status == DetectionHealthStatus.UNAVAILABLE
    assert worker.status_message == "RF-DETR model unavailable"


def test_failed_rfdetr_model_load_uses_safe_message(monkeypatch) -> None:
    worker = DetectionWorker()
    worker._session = MagicMock()
    monkeypatch.setattr("config.setting.RFDETR_MODEL_PATH", "/tmp/secret/model.pt")
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    with patch("core.infer.rfdetr_detector.RFDETRDetector.load", side_effect=RuntimeError("raw /tmp/secret traceback")):
        worker._load_detector()

    assert worker._detector is None
    assert worker._session.detection_status == DetectionHealthStatus.UNAVAILABLE
    assert worker.status_message == "RF-DETR model unavailable"
    assert "/tmp/secret" not in worker.status_message


def test_model_returned_classes_are_not_whitelisted() -> None:
    worker = DetectionWorker()
    worker._session = MagicMock()
    detection = Detection(
        class_name="shrimp",
        class_index=9,
        confidence=0.91,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_w=0.2,
        bbox_h=0.2,
    )

    result = worker._validated_detections([detection])

    assert result == [detection]


def test_missing_stream_frames_sets_detection_source_unavailable() -> None:
    worker = DetectionWorker()
    worker._session = MagicMock()
    worker._session.stream_status = StreamStatus.STREAM_UNAVAILABLE
    worker._detector = object()

    worker._sync_detection_state_for_frame_source(has_frame=False)

    assert worker._session.detection_status == DetectionHealthStatus.UNAVAILABLE
    assert worker.status_message == "Detection stream unavailable"


def test_duplicate_frame_id_is_not_processed_twice() -> None:
    worker = DetectionWorker()

    assert worker._should_process_frame(7) is True
    assert worker._should_process_frame(7) is False
    assert worker._should_process_frame(8) is True


def test_duplicate_frame_refreshes_stale_detection_status() -> None:
    now = datetime.now(timezone.utc)
    worker = DetectionWorker()
    worker._detector = object()
    worker._session = LiveStreamSession(
        session_id="session-1",
        source_id="cam",
        started_at=now - timedelta(seconds=10),
        stream_status=StreamStatus.LIVE,
        detection_status=DetectionHealthStatus.ACTIVE,
        last_frame_at=now,
        last_overlay_at=now - timedelta(seconds=4),
    )

    worker._refresh_detection_state(now)

    assert worker._session.detection_status == DetectionHealthStatus.STALE


def test_detection_fps_uses_overlay_intervals_not_inference_duration() -> None:
    now = datetime.now(timezone.utc)
    worker = DetectionWorker()
    worker._session = LiveStreamSession(
        session_id="session-1",
        source_id="cam",
        started_at=now,
    )

    worker._record_overlay_timing(10.0)
    worker._record_overlay_timing(10.1)

    assert worker._session.detection_fps == 10.0


def test_runtime_metrics_reset_between_worker_sessions() -> None:
    worker = DetectionWorker()
    worker._frame_counter = 10
    worker._overlay_intervals = [0.2]
    worker._last_overlay_monotonic = 123.0
    worker._last_processed_frame_id = 42

    worker._reset_runtime_metrics()

    assert worker._frame_counter == 0
    assert worker._overlay_intervals == []
    assert worker._last_overlay_monotonic is None
    assert worker._last_processed_frame_id is None
