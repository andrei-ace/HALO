"""Tests for halo.testing.EventRecorder."""

import asyncio

import pytest

from halo.contracts.events import EventEnvelope, EventType
from halo.runtime.event_bus import EventBus
from halo.testing.event_recorder import EventRecorder

ARM = "arm0"


def _make_event(bus: EventBus, event_type: EventType, arm_id: str = ARM) -> EventEnvelope:
    return EventEnvelope(
        event_id=bus.make_event_id(),
        type=event_type,
        ts_ms=0,
        arm_id=arm_id,
    )


@pytest.fixture
def bus() -> EventBus:
    b = EventBus()
    b._ensure_arm(ARM)
    return b


@pytest.fixture
async def recorder(bus: EventBus):
    rec = EventRecorder(bus, ARM)
    await rec.start()
    yield rec
    await rec.stop()


# -- basic recording -------------------------------------------------------


async def test_records_published_events(bus: EventBus, recorder: EventRecorder):
    evt = _make_event(bus, EventType.SKILL_STARTED)
    await bus.publish(evt)
    await asyncio.sleep(0.01)

    assert len(recorder.all_events) == 1
    assert recorder.all_events[0].event is evt
    assert recorder.all_events[0].seq == 1


async def test_records_multiple_events_in_order(bus: EventBus, recorder: EventRecorder):
    e1 = _make_event(bus, EventType.SKILL_STARTED)
    e2 = _make_event(bus, EventType.PHASE_ENTER)
    e3 = _make_event(bus, EventType.SKILL_SUCCEEDED)
    await bus.publish(e1)
    await bus.publish(e2)
    await bus.publish(e3)
    await asyncio.sleep(0.01)

    assert len(recorder.all_events) == 3
    assert recorder.event_types() == [EventType.SKILL_STARTED, EventType.PHASE_ENTER, EventType.SKILL_SUCCEEDED]
    assert recorder.all_events[0].seq == 1
    assert recorder.all_events[2].seq == 3


# -- filtering -------------------------------------------------------------


async def test_events_of_type_filters(bus: EventBus, recorder: EventRecorder):
    await bus.publish(_make_event(bus, EventType.SKILL_STARTED))
    await bus.publish(_make_event(bus, EventType.PHASE_ENTER))
    await bus.publish(_make_event(bus, EventType.SKILL_STARTED))
    await asyncio.sleep(0.01)

    starts = recorder.events_of_type(EventType.SKILL_STARTED)
    assert len(starts) == 2

    enters = recorder.events_of_type(EventType.PHASE_ENTER)
    assert len(enters) == 1


async def test_events_of_type_multiple_types(bus: EventBus, recorder: EventRecorder):
    await bus.publish(_make_event(bus, EventType.SKILL_STARTED))
    await bus.publish(_make_event(bus, EventType.PHASE_ENTER))
    await bus.publish(_make_event(bus, EventType.SKILL_SUCCEEDED))
    await asyncio.sleep(0.01)

    result = recorder.events_of_type(EventType.SKILL_STARTED, EventType.SKILL_SUCCEEDED)
    assert len(result) == 2


async def test_event_types_returns_sequence(bus: EventBus, recorder: EventRecorder):
    await bus.publish(_make_event(bus, EventType.PHASE_ENTER))
    await bus.publish(_make_event(bus, EventType.PHASE_EXIT))
    await asyncio.sleep(0.01)

    assert recorder.event_types() == [EventType.PHASE_ENTER, EventType.PHASE_EXIT]


# -- clear -----------------------------------------------------------------


async def test_clear_resets_events_and_seq(bus: EventBus, recorder: EventRecorder):
    await bus.publish(_make_event(bus, EventType.SKILL_STARTED))
    await asyncio.sleep(0.01)
    assert len(recorder.all_events) == 1

    recorder.clear()
    assert len(recorder.all_events) == 0

    await bus.publish(_make_event(bus, EventType.PHASE_ENTER))
    await asyncio.sleep(0.01)
    assert recorder.all_events[0].seq == 1  # seq restarted


# -- wait_for_event --------------------------------------------------------


async def test_wait_for_event_returns_existing(bus: EventBus, recorder: EventRecorder):
    await bus.publish(_make_event(bus, EventType.SKILL_SUCCEEDED))
    await asyncio.sleep(0.01)

    rec = await recorder.wait_for_event(EventType.SKILL_SUCCEEDED, timeout=1.0)
    assert rec.event.type == EventType.SKILL_SUCCEEDED


async def test_wait_for_event_blocks_until_arrival(bus: EventBus, recorder: EventRecorder):
    async def publish_later():
        await asyncio.sleep(0.05)
        await bus.publish(_make_event(bus, EventType.SKILL_FAILED))

    asyncio.create_task(publish_later())
    rec = await recorder.wait_for_event(EventType.SKILL_FAILED, timeout=2.0)
    assert rec.event.type == EventType.SKILL_FAILED


async def test_wait_for_event_timeout(bus: EventBus, recorder: EventRecorder):
    with pytest.raises(asyncio.TimeoutError):
        await recorder.wait_for_event(EventType.SAFETY_REFLEX_TRIGGERED, timeout=0.05)


# -- lifecycle -------------------------------------------------------------


async def test_stop_is_idempotent(bus: EventBus, recorder: EventRecorder):
    await recorder.stop()
    await recorder.stop()  # should not raise


async def test_recorded_at_is_monotonic(bus: EventBus, recorder: EventRecorder):
    for _ in range(5):
        await bus.publish(_make_event(bus, EventType.PHASE_ENTER))
    await asyncio.sleep(0.01)

    times = [r.recorded_at for r in recorder.all_events]
    assert times == sorted(times)
