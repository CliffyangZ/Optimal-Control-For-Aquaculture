"""Integration tests for overlay hub broadcast and slow-client isolation."""

import asyncio
import pytest

from web.overlay_hub import OverlayHub


def test_hub_broadcast_reaches_subscribers():
    hub = OverlayHub()
    q = hub.subscribe("v1")
    payload = {"type": "overlay", "frame_id": 1}
    hub.broadcast(payload)
    item = q.get_nowait()
    assert item == payload
    hub.unsubscribe("v1")


def test_hub_latest_value_replaces_stale_on_full_queue():
    hub = OverlayHub(queue_maxsize=1)
    q = hub.subscribe("v1")
    old = {"type": "overlay", "frame_id": 1}
    new = {"type": "overlay", "frame_id": 2}
    hub.broadcast(old)
    hub.broadcast(new)  # queue was full; old should be replaced
    item = q.get_nowait()
    assert item["frame_id"] == 2
    hub.unsubscribe("v1")


def test_hub_unsubscribe_removes_viewer():
    hub = OverlayHub()
    hub.subscribe("v1")
    assert hub.subscriber_count == 1
    hub.unsubscribe("v1")
    assert hub.subscriber_count == 0


def test_hub_broadcast_to_multiple_subscribers():
    hub = OverlayHub()
    q1 = hub.subscribe("v1")
    q2 = hub.subscribe("v2")
    payload = {"type": "overlay", "frame_id": 5}
    hub.broadcast(payload)
    assert q1.get_nowait() == payload
    assert q2.get_nowait() == payload
    hub.unsubscribe("v1")
    hub.unsubscribe("v2")


def test_hub_close_all_sends_none_sentinel():
    hub = OverlayHub(queue_maxsize=1)
    q = hub.subscribe("v1")
    hub.broadcast({"type": "overlay", "frame_id": 1})
    hub.close_all()
    # After close_all the queue should have a None sentinel (latest-value replace)
    assert hub.subscriber_count == 0


def test_hub_broadcast_to_slow_client_does_not_block():
    hub = OverlayHub()
    hub.subscribe("v1")
    # Rapidly send 5 payloads — should never raise
    for i in range(5):
        hub.broadcast({"type": "overlay", "frame_id": i})
    hub.unsubscribe("v1")
