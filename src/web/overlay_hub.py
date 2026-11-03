"""Synchronous thread-safe broadcast hub for WebSocket overlay messages.

Subscribers (viewers) get their own queue.Queue. On broadcast, payloads go to all
queues with latest-value semantics: if a queue is full, drop the oldest item to
make room for the newest.
"""

from __future__ import annotations

import queue
from typing import Optional


class OverlayHub:
    """Thread-safe broadcast hub with per-viewer queues and latest-value replacement."""

    def __init__(self, queue_maxsize: int = 5) -> None:
        self._queues: dict[str, queue.Queue] = {}
        self._queue_maxsize = queue_maxsize

    def subscribe(self, viewer_id: str) -> queue.Queue:
        """Subscribe a viewer and return their queue.

        If viewer_id already subscribed, returns the existing queue.
        """
        if viewer_id not in self._queues:
            self._queues[viewer_id] = queue.Queue(maxsize=self._queue_maxsize)
        return self._queues[viewer_id]

    def unsubscribe(self, viewer_id: str) -> None:
        """Unsubscribe a viewer."""
        if viewer_id in self._queues:
            del self._queues[viewer_id]

    def broadcast(self, payload: dict) -> None:
        """Broadcast payload to all subscriber queues.

        Uses latest-value semantics: if a queue is full, drops the oldest item
        to make room for the newest. This method never blocks.
        """
        for q in self._queues.values():
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    def close_all(self) -> None:
        """Send None sentinel to all queues to signal disconnect, then clear."""
        for q in self._queues.values():
            try:
                q.put_nowait(None)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(None)
                except queue.Empty:
                    pass
        self._queues.clear()

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return len(self._queues)


overlay_hub: OverlayHub = OverlayHub()
