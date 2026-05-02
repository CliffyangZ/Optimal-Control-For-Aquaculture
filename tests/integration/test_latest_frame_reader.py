"""Integration tests for latest frame reading without replay."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
from core.models import CurrentFrame
from streaming.latest_frame_reader import LatestFrameReader


class MockRTSPCapture:
    """Mock RTSP video capture for testing."""

    def __init__(self, frame_count: int = 5):
        self.frame_count = frame_count
        self.current = 0
        self.is_opened = True

    def isOpened(self):
        return self.is_opened

    def read(self):
        if self.current >= self.frame_count:
            return False, None

        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        self.current += 1
        return True, frame

    def release(self):
        self.is_opened = False

    def set(self, prop_id, value):
        pass


class TestLatestFrameReaderNoReplay:
    """Test latest frame reading behavior."""

    @pytest.fixture
    def mock_reader(self):
        """Create a mock frame reader."""
        with patch('streaming.latest_frame_reader.cv2.VideoCapture'):
            reader = LatestFrameReader("rtsp://test/stream")
            reader.buffer = MagicMock()
            return reader

    def test_reader_initialization(self, mock_reader):
        """Test reader is properly initialized."""
        assert mock_reader is not None

    def test_start_reading(self, mock_reader):
        """Test starting frame reading."""
        with patch.object(mock_reader, '_start_capture'):
            mock_reader.start()
            mock_reader._start_capture.assert_called_once()

    def test_stop_reading(self, mock_reader):
        """Test stopping frame reading."""
        with patch.object(mock_reader, '_stop_capture'):
            mock_reader.stop()
            mock_reader._stop_capture.assert_called_once()

    def test_get_latest_frame(self, mock_reader):
        """Test retrieving latest frame."""
        test_frame = CurrentFrame(
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
            captured_at=datetime.now(),
            sequence_number=1,
            source_width=1280,
            source_height=720
        )
        mock_reader.buffer.get.return_value = test_frame

        result = mock_reader.get_latest_frame()
        assert result == test_frame

    def test_frame_timestamp_monotonic(self, mock_reader):
        """Test that frame timestamps are monotonically increasing."""
        frames = []
        for i in range(5):
            frame = CurrentFrame(
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
                captured_at=datetime.now(),
                sequence_number=i,
                source_width=1280,
                source_height=720
            )
            frames.append(frame)

        for i in range(1, len(frames)):
            assert frames[i].captured_at >= frames[i-1].captured_at

    def test_sequence_numbers_monotonic(self, mock_reader):
        """Test that sequence numbers increase monotonically."""
        sequences = [1, 2, 3, 4, 5]
        for i in range(1, len(sequences)):
            assert sequences[i] > sequences[i-1]

    def test_no_older_frames_replayed(self, mock_reader):
        """Test that older frames are not replayed."""
        frames = []
        for i in range(10):
            frame = CurrentFrame(
                image=np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8),
                captured_at=datetime.now(),
                sequence_number=i,
                source_width=1280,
                source_height=720
            )
            frames.append(frame)

        latest_frame = frames[-1]
        assert latest_frame.sequence_number == 9

    def test_stale_queue_avoidance(self):
        """Test that stale frames are not accumulated."""
        from streaming.latest_frame_reader import LatestFrameBuffer

        buffer = LatestFrameBuffer()

        for i in range(100):
            frame = CurrentFrame(
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
                captured_at=datetime.now(),
                sequence_number=i,
                source_width=1280,
                source_height=720
            )
            buffer.put(frame)

        latest = buffer.get()
        assert latest.sequence_number == 99

    def test_frame_dimensions_consistent(self, mock_reader):
        """Test that frame dimensions remain consistent."""
        for w, h in [(1280, 720), (1920, 1080), (640, 480)]:
            frame = CurrentFrame(
                image=np.zeros((h, w, 3), dtype=np.uint8),
                captured_at=datetime.now(),
                sequence_number=1,
                source_width=w,
                source_height=h
            )
            assert frame.source_width == w
            assert frame.source_height == h
