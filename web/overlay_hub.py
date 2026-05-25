"""Latest-value broadcast hub for overlay WebSocket subscribers."""

from __future__ import annotations

import asyncio

from config.setting import OVERLAY_HUB_QUEUE_MAXSIZE


class OverlayHub:
    def __init__(self, queue_maxsize: int = OVERLAY_HUB_QUEUE_MAXSIZE) -> None:
        self._queue_maxsize = queue_maxsize
        self._subscribers: dict[str, asyncio.Queue] = {}

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self, viewer_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers[viewer_id] = queue
        return queue

    def unsubscribe(self, viewer_id: str) -> None:
        self._subscribers.pop(viewer_id, None)

    def broadcast(self, payload: dict | None) -> None:
        for queue in list(self._subscribers.values()):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def close_all(self) -> None:
        for queue in list(self._subscribers.values()):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self._subscribers.clear()


overlay_hub = OverlayHub()
