"""Tests for EventBus."""

import asyncio

import pytest

from halo.contracts.events import EventEnvelope, EventType
from halo.runtime.event_bus import EventBus

ARM = "arm0"
ARM2 = "arm1"


def _evt(arm_id: str = ARM, event_id: str = "evt-1", ts_ms: int = 1000) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        type=EventType.PHASE_ENTER,
        ts_ms=ts_ms,
        arm_id=arm_id,
        data={"phase": "SELECT_GRASP"},
    )


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


# ---------------------------------------------------------------------------
# Basic pub/sub
# ---------------------------------------------------------------------------


async def test_subscriber_receives_event(bus: EventBus):
    q = bus.subscribe(ARM)
    evt = _evt()
    await bus.publish(evt)
    received = await asyncio.wait_for(q.get(), timeout=1.0)
    assert received == evt


async def test_multiple_subscribers_all_receive(bus: EventBus):
    q1 = bus.subscribe(ARM)
    q2 = bus.subscribe(ARM)
    evt = _evt()
    await bus.publish(evt)
    r1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    r2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert r1 == evt
    assert r2 == evt


async def test_unsubscribe_stops_delivery(bus: EventBus):
    q = bus.subscribe(ARM)
    bus.unsubscribe(ARM, q)
    await bus.publish(_evt())
    assert q.empty()


async def test_unsubscribe_unknown_queue_does_not_raise(bus: EventBus):
    orphan: asyncio.Queue[EventEnvelope] = asyncio.Queue()
    bus.unsubscribe(ARM, orphan)  # should not raise


# ---------------------------------------------------------------------------
# Arm isolation
# ---------------------------------------------------------------------------


async def test_events_only_reach_correct_arm_subscribers(bus: EventBus):
    q0 = bus.subscribe(ARM)
    q1 = bus.subscribe(ARM2)

    await bus.publish(_evt(arm_id=ARM))
    await bus.publish(_evt(arm_id=ARM2, event_id="evt-2"))

    r0 = await asyncio.wait_for(q0.get(), timeout=1.0)
    r1 = await asyncio.wait_for(q1.get(), timeout=1.0)

    assert r0.arm_id == ARM
    assert r1.arm_id == ARM2
    assert q0.empty()
    assert q1.empty()


# ---------------------------------------------------------------------------
# Recent events ring
# ---------------------------------------------------------------------------


async def test_get_recent_events_empty_before_publish(bus: EventBus):
    assert bus.get_recent_events(ARM) == []


async def test_recent_events_ring_limited(bus: EventBus):
    bus.subscribe(ARM)  # ensure arm is registered
    for i in range(EventBus.RECENT_EVENTS_RING_SIZE + 4):
        await bus.publish(_evt(event_id=f"evt-{i}", ts_ms=i))

    recent = bus.get_recent_events(ARM)
    assert len(recent) == EventBus.RECENT_EVENTS_RING_SIZE
    # most recent event should be last
    assert recent[-1].event_id == f"evt-{EventBus.RECENT_EVENTS_RING_SIZE + 3}"


async def test_get_recent_events_n_param(bus: EventBus):
    bus.subscribe(ARM)
    for i in range(6):
        await bus.publish(_evt(event_id=f"evt-{i}"))
    assert len(bus.get_recent_events(ARM, n=3)) == 3


async def test_recent_events_arm_isolation(bus: EventBus):
    bus.subscribe(ARM)
    bus.subscribe(ARM2)
    await bus.publish(_evt(arm_id=ARM, event_id="a"))
    await bus.publish(_evt(arm_id=ARM2, event_id="b"))

    assert [e.event_id for e in bus.get_recent_events(ARM)] == ["a"]
    assert [e.event_id for e in bus.get_recent_events(ARM2)] == ["b"]


# ---------------------------------------------------------------------------
# Full queue — non-blocking drop
# ---------------------------------------------------------------------------


async def test_full_subscriber_queue_does_not_block(bus: EventBus):
    q = bus.subscribe(ARM, maxsize=2)
    # Fill the queue
    await bus.publish(_evt(event_id="evt-1"))
    await bus.publish(_evt(event_id="evt-2"))
    # This third publish must not raise or block
    await bus.publish(_evt(event_id="evt-3"))
    assert q.qsize() == 2  # third event was dropped


# ---------------------------------------------------------------------------
# Event ID generation
# ---------------------------------------------------------------------------


def test_make_event_id_is_unique(bus: EventBus):
    ids = [bus.make_event_id() for _ in range(100)]
    assert len(set(ids)) == 100


def test_make_event_id_format(bus: EventBus):
    eid = bus.make_event_id()
    assert eid.startswith("evt-")
