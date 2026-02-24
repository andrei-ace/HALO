"""Tests for HALORuntime: get_latest_runtime_snapshot wires store + bus."""

import pytest

from halo.contracts.enums import PhaseId, SkillName
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import SkillInfo
from halo.runtime.runtime import HALORuntime

ARM = "arm0"


@pytest.fixture
def rt() -> HALORuntime:
    r = HALORuntime()
    r.register_arm(ARM)
    return r


async def test_returns_snapshot_with_correct_arm(rt: HALORuntime):
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.arm_id == ARM
    assert snap.snapshot_id.startswith(f"snap-{ARM}-")


async def test_snapshot_includes_recent_bus_events(rt: HALORuntime):
    rt.bus.subscribe(ARM)  # ensure arm is registered in bus
    evt = EventEnvelope(event_id="evt-1", type=EventType.PHASE_ENTER, ts_ms=1000, arm_id=ARM)
    await rt.bus.publish(evt)
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert any(e.event_id == "evt-1" for e in snap.recent_events)


async def test_snapshot_reflects_store_updates(rt: HALORuntime):
    skill = SkillInfo(name=SkillName.PICK, skill_run_id="run-1", phase=PhaseId.LIFT)
    await rt.store.update_skill(ARM, skill)
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill == skill


async def test_each_call_replaces_previous_snapshot(rt: HALORuntime):
    snap1 = await rt.get_latest_runtime_snapshot(ARM)
    snap2 = await rt.get_latest_runtime_snapshot(ARM)
    assert snap1.snapshot_id != snap2.snapshot_id
    # store must hold snap2, not snap1
    cached = await rt.store.get_latest_snapshot(ARM)
    assert cached == snap2


async def test_two_arms_are_independent(rt: HALORuntime):
    rt.register_arm("arm1")
    snap0 = await rt.get_latest_runtime_snapshot(ARM)
    snap1 = await rt.get_latest_runtime_snapshot("arm1")
    assert snap0.arm_id == ARM
    assert snap1.arm_id == "arm1"
    assert snap0.snapshot_id != snap1.snapshot_id
