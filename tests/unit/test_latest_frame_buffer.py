"""Tests for latest-frame buffer with sequence number tracking."""

import pytest
from datetime import datetime
import numpy as np
from core.models import CurrentFrame
from streaming.latest_frame_reader import LatestFrameBuffer


class TestLatestFrameBuffer:
    """Test thread-safe latest-frame buffer implementation."""

    @pytest.fixture
    def buffer(self):
        """Create a fresh frame buffer."""
        return LatestFrameBuffer()

    def create_test_frame(self, seq_num: int, width: int = 1280, height: int = 720):
        """Helper to create a test frame."""
        return CurrentFrame(
            image=np.zeros((height, width, 3), dtype=np.uint8),
            captured_at=datetime.now(),
            sequence_number=seq_num,
            source_width=width,
            source_height=height
        )

    def test_put_and_get_single_frame(self, buffer):
        """Test adding and retrieving a single frame."""
        frame = self.create_test_frame(1)
        buffer.put(frame)
        retrieved = buffer.get()
        assert retrieved is not None
        assert retrieved.sequence_number == 1

    def test_latest_frame_replacement(self, buffer):
        """Test that newer frames replace older ones."""
        frame1 = self.create_test_frame(1)
        frame2 = self.create_test_frame(2)
        frame3 = self.create_test_frame(3)

        buffer.put(frame1)
        buffer.put(frame2)
        buffer.put(frame3)

        retrieved = buffer.get()
        assert retrieved.sequence_number == 3

    def test_no_stale_queue_buildup(self, buffer):
        """Test that old frames don't accumulate in queue."""
        for i in range(1, 11):
            frame = self.create_test_frame(i)
            buffer.put(frame)

        retrieved = buffer.get()
        assert retrieved.sequence_number == 10

    def test_multiple_reads_same_frame(self, buffer):
        """Test that multiple reads return the same latest frame."""
        frame = self.create_test_frame(1)
        buffer.put(frame)

        first_read = buffer.get()
        second_read = buffer.get()

        assert first_read.sequence_number == second_read.sequence_number == 1

    def test_buffer_empty_returns_none(self, buffer):
        """Test that empty buffer returns None."""
        result = buffer.get()
        assert result is None

    def test_sequence_numbers_monotonic(self, buffer):
        """Test that sequence numbers are preserved correctly."""
        frames = [self.create_test_frame(i) for i in range(1, 6)]
        for frame in frames:
            buffer.put(frame)

        retrieved = buffer.get()
        assert retrieved.sequence_number == 5

    def test_frame_dimensions_preserved(self, buffer):
        """Test that frame dimensions are preserved."""
        frame = self.create_test_frame(1, width=1920, height=1080)
        buffer.put(frame)
        retrieved = buffer.get()

        assert retrieved.source_width == 1920
        assert retrieved.source_height == 1080

    def test_rapid_updates(self, buffer):
        """Test rapid sequential updates."""
        for i in range(100):
            frame = self.create_test_frame(i)
            buffer.put(frame)

        retrieved = buffer.get()
        assert retrieved.sequence_number == 99
