from __future__ import annotations

import asyncio
import itertools
from collections import deque

from halo.contracts.events import EventEnvelope


class EventBus:
    """
    Asyncio pub/sub event bus, partitioned by arm_id.

    Each subscriber receives its own asyncio.Queue. Publishing is non-blocking:
    if a subscriber queue is full the event is silently dropped for that subscriber
    rather than blocking the publisher. A per-arm ring of recent events is kept
    for snapshot assembly.
    """

    RECENT_EVENTS_RING_SIZE = 8

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[EventEnvelope]]] = {}
        self._recent: dict[str, deque[EventEnvelope]] = {}
        self._lock = asyncio.Lock()
        self._counter = itertools.count(1)

    def make_event_id(self) -> str:
        return f"evt-{next(self._counter)}"

    def _ensure_arm(self, arm_id: str) -> None:
        if arm_id not in self._subscribers:
            self._subscribers[arm_id] = []
            self._recent[arm_id] = deque(maxlen=self.RECENT_EVENTS_RING_SIZE)

    def subscribe(self, arm_id: str, maxsize: int = 100) -> asyncio.Queue[EventEnvelope]:
        """Return a new queue that will receive all future events for arm_id."""
        self._ensure_arm(arm_id)
        q: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=maxsize)
        self._subscribers[arm_id].append(q)
        return q

    def unsubscribe(self, arm_id: str, queue: asyncio.Queue[EventEnvelope]) -> None:
        """Remove a previously subscribed queue."""
        if arm_id in self._subscribers:
            try:
                self._subscribers[arm_id].remove(queue)
            except ValueError:
                pass

    async def publish(self, event: EventEnvelope) -> None:
        """Append event to the recent ring and deliver to all subscribers (non-blocking)."""
        async with self._lock:
            self._ensure_arm(event.arm_id)
            self._recent[event.arm_id].append(event)
            for q in self._subscribers[event.arm_id]:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # slow subscriber — drop rather than block

    def get_recent_events(self, arm_id: str, n: int = RECENT_EVENTS_RING_SIZE) -> list[EventEnvelope]:
        """Return up to n most recent events for arm_id (oldest first)."""
        ring = self._recent.get(arm_id)
        if ring is None:
            return []
        events = list(ring)
        return events[-n:] if n < len(events) else events
