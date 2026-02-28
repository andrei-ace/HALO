"""EventRecorder — subscribes to EventBus and records all events for test assertions."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from halo.contracts.events import EventEnvelope, EventType
from halo.runtime.event_bus import EventBus


@dataclass(frozen=True)
class RecordedEvent:
    """An event captured by EventRecorder with metadata for analysis."""

    event: EventEnvelope
    recorded_at: float  # time.monotonic() when recorded
    seq: int  # monotonically increasing sequence number


class EventRecorder:
    """Subscribe to an EventBus and record all events for post-hoc analysis.

    Uses an unbounded queue so events are never dropped. Provides query helpers
    for filtering by type and an async ``wait_for_event`` for test assertions.

    Lifecycle::

        recorder = EventRecorder(bus, arm_id)
        await recorder.start()
        # ... run test ...
        assert len(recorder.events_of_type(EventType.SKILL_SUCCEEDED)) == 1
        await recorder.stop()
    """

    def __init__(self, bus: EventBus, arm_id: str) -> None:
        self._bus = bus
        self._arm_id = arm_id
        self._queue: asyncio.Queue[EventEnvelope] | None = None
        self._events: list[RecordedEvent] = []
        self._seq = 0
        self._drain_task: asyncio.Task | None = None
        self._waiters: dict[EventType, list[asyncio.Future[RecordedEvent]]] = {}

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to the bus and start draining events."""
        self._queue = self._bus.subscribe(self._arm_id, maxsize=0)  # unbounded
        self._drain_task = asyncio.create_task(self._drain(), name="event-recorder-drain")

    async def stop(self) -> None:
        """Cancel the drain task and unsubscribe."""
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None
        if self._queue is not None:
            self._bus.unsubscribe(self._arm_id, self._queue)
            self._queue = None

    # -- drain loop --------------------------------------------------------

    async def _drain(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            self._seq += 1
            rec = RecordedEvent(event=event, recorded_at=time.monotonic(), seq=self._seq)
            self._events.append(rec)
            # Wake any futures waiting for this event type
            waiters = self._waiters.get(event.type)
            if waiters:
                for fut in waiters:
                    if not fut.done():
                        fut.set_result(rec)
                self._waiters[event.type] = []

    # -- query API ---------------------------------------------------------

    @property
    def all_events(self) -> list[RecordedEvent]:
        """Return all recorded events in order."""
        return list(self._events)

    def events_of_type(self, *types: EventType) -> list[RecordedEvent]:
        """Return recorded events matching any of the given types."""
        type_set = set(types)
        return [r for r in self._events if r.event.type in type_set]

    def event_types(self) -> list[EventType]:
        """Return the sequence of event types in order."""
        return [r.event.type for r in self._events]

    def clear(self) -> None:
        """Discard all recorded events and reset the sequence counter."""
        self._events.clear()
        self._seq = 0

    async def wait_for_event(self, event_type: EventType, timeout: float = 5.0) -> RecordedEvent:
        """Wait until an event of the given type is recorded.

        If an event of this type was already recorded since the last ``clear()``,
        returns immediately with the most recent one.

        Raises ``asyncio.TimeoutError`` if no matching event arrives within *timeout*.
        """
        # Check if already present
        existing = self.events_of_type(event_type)
        if existing:
            return existing[-1]

        # Register a future
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[RecordedEvent] = loop.create_future()
        self._waiters.setdefault(event_type, []).append(fut)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            # Clean up the future from the waiters list
            waiters = self._waiters.get(event_type, [])
            if fut in waiters:
                waiters.remove(fut)
            raise
