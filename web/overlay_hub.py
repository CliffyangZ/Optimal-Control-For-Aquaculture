"""In-memory latest-value WebSocket subscriber hub."""

import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OverlayHub:
    """Broadcasts the latest overlay payload to all connected WebSocket clients.

    Uses latest-value semantics: slow clients get the newest message, not a backlog.
    Each subscriber gets a bounded asyncio.Queue(maxsize=1). If the queue is full
    the oldest item is discarded before the new one is added so detection is never blocked.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue[Optional[dict]]] = {}

    def subscribe(self, viewer_id: str) -> asyncio.Queue[Optional[dict]]:
        q: asyncio.Queue[Optional[dict]] = asyncio.Queue(maxsize=1)
        self._subscribers[viewer_id] = q
        logger.debug("overlay_hub: subscribed viewer=%s total=%d", viewer_id, len(self._subscribers))
        return q

    def unsubscribe(self, viewer_id: str) -> None:
        self._subscribers.pop(viewer_id, None)
        logger.debug("overlay_hub: unsubscribed viewer=%s total=%d", viewer_id, len(self._subscribers))

    def broadcast(self, payload: dict) -> None:
        """Publish payload to all subscribers without blocking."""
        for viewer_id, q in list(self._subscribers.items()):
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(payload)
            except Exception:
                logger.warning("overlay_hub: failed to enqueue for viewer=%s", viewer_id)

    def close_all(self) -> None:
        """Signal all subscribers to stop by sending None sentinel."""
        for q in self._subscribers.values():
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(None)
            except Exception:
                pass
        self._subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


overlay_hub = OverlayHub()
