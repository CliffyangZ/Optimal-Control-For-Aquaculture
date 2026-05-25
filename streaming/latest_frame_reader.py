"""RTSP frame acquisition with latest-frame-only buffering."""

import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime, timezone
from core.models import CurrentFrame
from config.setting import INTERRUPTION_TIMEOUT_SECONDS


class LatestFrameBuffer:
    """Thread-safe buffer that holds only the latest frame."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, frame: CurrentFrame) -> None:
        """Add a frame to the buffer, replacing any older frame."""
        with self._lock:
            self._frame = frame

    def get(self) -> CurrentFrame:
        """Get the latest frame without removing it."""
        with self._lock:
            return self._frame


class LatestFrameReader:
    """Reads RTSP stream with latest-frame-only acquisition."""

    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.buffer = LatestFrameBuffer()
        self._capture = None
        self._reading = False
        self._reader_thread = None
        self._sequence_number = 0
        self._last_frame_time = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 30

    def start(self) -> None:
        """Start reading frames from RTSP stream."""
        if self._reading:
            return

        self._reading = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        """Stop reading frames and release resources."""
        self._reading = False
        # Release capture before join: unblocks a thread stuck in capture.read()
        self._stop_capture()
        if self._reader_thread:
            self._reader_thread.join(timeout=2)

    def get_latest_frame(self) -> CurrentFrame:
        """Get the latest frame if available."""
        return self.buffer.get()

    def get_last_frame_time(self) -> datetime:
        """Get the timestamp of the last received frame."""
        return self._last_frame_time

    def _read_loop(self) -> None:
        """Background thread that continuously reads frames."""
        self._start_capture()
        self._consecutive_failures = 0

        while self._reading:
            if self._capture is None or not self._capture.isOpened():
                self._consecutive_failures += 1
                if self._consecutive_failures > self._max_consecutive_failures:
                    self._stop_capture()
                    self._start_capture()
                    self._consecutive_failures = 0
                time.sleep(0.1)
                continue

            ret, frame = self._capture.read()
            if not ret:
                self._consecutive_failures += 1
                if self._consecutive_failures > 10:
                    self._stop_capture()
                    self._start_capture()
                    self._consecutive_failures = 0
                time.sleep(0.1)
                continue

            self._consecutive_failures = 0
            self._sequence_number += 1
            self._last_frame_time = datetime.now(timezone.utc)
            source_height, source_width = frame.shape[:2]

            current_frame = CurrentFrame(
                image=frame,
                captured_at=self._last_frame_time,
                sequence_number=self._sequence_number,
                source_width=source_width,
                source_height=source_height
            )
            self.buffer.put(current_frame)

    def _start_capture(self) -> None:
        """Open RTSP connection with optimized buffer settings."""
        try:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
            )
            self._capture = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            self._capture = None

    def _stop_capture(self) -> None:
        """Close RTSP connection."""
        if self._capture:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
