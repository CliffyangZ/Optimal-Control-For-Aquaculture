"""WebSocket subscription and broadcast coordination primitives."""

import asyncio

from pi_controller.schemas import WebSocketEvent


class RealtimeHub:
    """Manages WebSocket subscriptions and broadcasts events to all connected clients."""

    def __init__(self) -> None:
        """Initialize the realtime hub."""
        self.subscriptions: dict[str, asyncio.Queue] = {}
        self.lock = asyncio.Lock()

    async def subscribe(self, client_id: str) -> asyncio.Queue:
        """Subscribe a client and return their message queue."""
        async with self.lock:
            if client_id not in self.subscriptions:
                self.subscriptions[client_id] = asyncio.Queue()
            return self.subscriptions[client_id]

    async def unsubscribe(self, client_id: str) -> None:
        """Unsubscribe a client."""
        async with self.lock:
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]

    async def broadcast(self, event: WebSocketEvent) -> None:
        """Broadcast an event to all subscribed clients."""
        message = event.model_dump_json()
        async with self.lock:
            for queue in self.subscriptions.values():
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    pass

    async def send_to_client(self, client_id: str, event: WebSocketEvent) -> bool:
        """Send an event to a specific client."""
        async with self.lock:
            queue = self.subscriptions.get(client_id)
            if not queue:
                return False
            try:
                queue.put_nowait(event.model_dump_json())
                return True
            except asyncio.QueueFull:
                return False

    async def client_count(self) -> int:
        """Get the number of connected clients."""
        async with self.lock:
            return len(self.subscriptions)
