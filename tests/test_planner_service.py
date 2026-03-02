"""Tests for PlannerService: tick, lifecycle, urgent events, and snapshot serializer."""

import asyncio
import time
import uuid

import pytest

from halo.contracts.commands import CommandEnvelope, StartSkillPayload
from halo.contracts.enums import (
    ActStatus,
    CommandAckStatus,
    CommandType,
    PerceptionFailureCode,
    PhaseId,
    SafetyState,
    SkillName,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import (
    ActInfo,
    OutcomeInfo,
    PerceptionInfo,
    PlannerSnapshot,
    ProgressInfo,
    SafetyInfo,
    SkillInfo,
    TargetInfo,
)
from halo.runtime.runtime import HALORuntime
from halo.services.planner_service.config import PlannerServiceConfig
from halo.services.planner_service.service import PlannerService
from halo.services.planner_service.snapshot_serializer import snapshot_to_dict
from halo.services.planner_service.tools import AgentContext, build_tools

ARM = "arm0"


@pytest.fixture
def rt() -> HALORuntime:
    r = HALORuntime()
    r.register_arm(ARM)
    return r


async def _null_decide(snap: PlannerSnapshot) -> list[CommandEnvelope]:
    return []


def _cmd(
    cmd_id: str,
    snap_id: str | None = None,
    cmd_type: CommandType = CommandType.START_SKILL,
) -> CommandEnvelope:
    return CommandEnvelope(
        command_id=cmd_id,
        arm_id=ARM,
        issued_at_ms=int(time.monotonic() * 1000),
        type=cmd_type,
        payload=StartSkillPayload(skill_name=SkillName.PICK, target_handle="obj-1"),
        precondition_snapshot_id=snap_id,
    )


# ─── tick() core ──────────────────────────────────────────────────────────────


async def test_tick_calls_decide_fn_with_snapshot(rt: HALORuntime):
    received: list[PlannerSnapshot] = []

    async def decide(snap: PlannerSnapshot) -> list[CommandEnvelope]:
        received.append(snap)
        return []

    svc = PlannerService(ARM, rt, decide)
    await svc.tick()

    assert len(received) == 1
    assert received[0].arm_id == ARM


async def test_tick_returns_empty_acks_when_no_commands(rt: HALORuntime):
    svc = PlannerService(ARM, rt, _null_decide)
    acks = await svc.tick()
    assert acks == []


async def test_tick_submits_start_skill_command(rt: HALORuntime):
    await rt.get_latest_runtime_snapshot(ARM)

    async def decide(s: PlannerSnapshot) -> list[CommandEnvelope]:
        return [_cmd(str(uuid.uuid4()), snap_id=s.snapshot_id)]

    svc = PlannerService(ARM, rt, decide)
    acks = await svc.tick()

    assert len(acks) == 1
    assert acks[0].status == CommandAckStatus.ACCEPTED


async def test_tick_returns_ack_for_each_command(rt: HALORuntime):
    await rt.get_latest_runtime_snapshot(ARM)

    async def decide(s: PlannerSnapshot) -> list[CommandEnvelope]:
        return [
            _cmd(str(uuid.uuid4()), snap_id=s.snapshot_id),
            _cmd(str(uuid.uuid4()), snap_id=s.snapshot_id),
        ]

    svc = PlannerService(ARM, rt, decide)
    acks = await svc.tick()

    assert len(acks) == 2
    assert all(a.status == CommandAckStatus.ACCEPTED for a in acks)


async def test_tick_gets_fresh_snapshot_each_call(rt: HALORuntime):
    seen_ids: list[str] = []

    async def decide(snap: PlannerSnapshot) -> list[CommandEnvelope]:
        seen_ids.append(snap.snapshot_id)
        return []

    svc = PlannerService(ARM, rt, decide)
    await svc.tick()
    await svc.tick()

    assert len(seen_ids) == 2
    assert seen_ids[0] != seen_ids[1]


async def test_tick_stale_precondition_returns_rejected_ack(rt: HALORuntime):
    async def decide(snap: PlannerSnapshot) -> list[CommandEnvelope]:
        return [_cmd(str(uuid.uuid4()), snap_id="bogus-snapshot-id")]

    svc = PlannerService(ARM, rt, decide)
    acks = await svc.tick()

    assert len(acks) == 1
    assert acks[0].status == CommandAckStatus.REJECTED_STALE


async def test_tick_duplicate_command_returns_already_applied(rt: HALORuntime):
    cmd_id = str(uuid.uuid4())
    await rt.get_latest_runtime_snapshot(ARM)

    call_count = 0

    async def decide(s: PlannerSnapshot) -> list[CommandEnvelope]:
        nonlocal call_count
        call_count += 1
        # Always issue the same command_id; first tick uses valid snap_id
        return [_cmd(cmd_id, snap_id=s.snapshot_id if call_count == 1 else None)]

    svc = PlannerService(ARM, rt, decide)
    acks1 = await svc.tick()
    acks2 = await svc.tick()

    assert acks1[0].status == CommandAckStatus.ACCEPTED
    assert acks2[0].status == CommandAckStatus.ALREADY_APPLIED


# ─── Config ───────────────────────────────────────────────────────────────────


async def test_tick_limits_commands_to_max_per_tick(rt: HALORuntime):
    async def decide(snap: PlannerSnapshot) -> list[CommandEnvelope]:
        return [_cmd(str(uuid.uuid4())) for _ in range(5)]

    cfg = PlannerServiceConfig(max_commands_per_tick=2)
    svc = PlannerService(ARM, rt, decide, config=cfg)
    acks = await svc.tick()

    assert len(acks) == 2


# ─── Lifecycle ────────────────────────────────────────────────────────────────


async def test_start_stop_lifecycle(rt: HALORuntime):
    svc = PlannerService(ARM, rt, _null_decide)

    assert svc._loop_task is None
    assert svc._event_task is None

    await svc.start()

    assert svc._loop_task is not None
    assert svc._event_task is not None

    await svc.stop()

    assert svc._loop_task is None
    assert svc._event_task is None
    assert svc._event_queue is None


async def test_start_subscribes_to_event_bus(rt: HALORuntime):
    svc = PlannerService(ARM, rt, _null_decide)
    assert svc._event_queue is None

    await svc.start()
    assert svc._event_queue is not None

    await svc.stop()


# ─── Urgent events ────────────────────────────────────────────────────────────


async def test_urgent_event_triggers_tick(rt: HALORuntime):
    call_count = 0

    async def decide(snap: PlannerSnapshot) -> list[CommandEnvelope]:
        nonlocal call_count
        call_count += 1
        return []

    # Long watchdog so it won't fire during the test; ticks only via events.
    cfg = PlannerServiceConfig(watchdog_interval_s=100.0)
    svc = PlannerService(ARM, rt, decide, config=cfg)
    await svc.start()

    # Loop is now waiting for an event — no tick yet.
    await asyncio.sleep(0.05)
    assert call_count == 0

    # Publish an urgent event — should wake the loop and trigger a tick.
    event = EventEnvelope(
        event_id=rt.bus.make_event_id(),
        type=EventType.SKILL_SUCCEEDED,
        ts_ms=int(time.monotonic() * 1000),
        arm_id=ARM,
        data={},
    )
    await rt.bus.publish(event)

    await asyncio.sleep(0.15)
    assert call_count == 1

    await svc.stop()


async def test_target_acquired_event_triggers_tick(rt: HALORuntime):
    call_count = 0

    async def decide(snap: PlannerSnapshot) -> list[CommandEnvelope]:
        nonlocal call_count
        call_count += 1
        return []

    cfg = PlannerServiceConfig(watchdog_interval_s=100.0)
    svc = PlannerService(ARM, rt, decide, config=cfg)
    await svc.start()

    await asyncio.sleep(0.05)
    assert call_count == 0

    event = EventEnvelope(
        event_id=rt.bus.make_event_id(),
        type=EventType.TARGET_ACQUIRED,
        ts_ms=int(time.monotonic() * 1000),
        arm_id=ARM,
        data={"target_handle": "cube-1"},
    )
    await rt.bus.publish(event)

    await asyncio.sleep(0.15)
    assert call_count == 1

    await svc.stop()


async def test_command_rejected_event_triggers_tick(rt: HALORuntime):
    call_count = 0

    async def decide(snap: PlannerSnapshot) -> list[CommandEnvelope]:
        nonlocal call_count
        call_count += 1
        return []

    cfg = PlannerServiceConfig(watchdog_interval_s=100.0)
    svc = PlannerService(ARM, rt, decide, config=cfg)
    await svc.start()

    await asyncio.sleep(0.05)
    assert call_count == 0

    event = EventEnvelope(
        event_id=rt.bus.make_event_id(),
        type=EventType.COMMAND_REJECTED,
        ts_ms=int(time.monotonic() * 1000),
        arm_id=ARM,
        data={"command_id": "cmd-1", "command_type": "START_SKILL"},
    )
    await rt.bus.publish(event)

    await asyncio.sleep(0.15)
    assert call_count == 1

    await svc.stop()


# ─── Snapshot serializer ──────────────────────────────────────────────────────


def _make_snapshot(
    skill: SkillInfo | None = None,
    target: TargetInfo | None = None,
    events: tuple = (),
    held_object_handle: str | None = None,
) -> PlannerSnapshot:
    return PlannerSnapshot(
        snapshot_id="snap-1",
        ts_ms=1000,
        arm_id=ARM,
        skill=skill,
        target=target,
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False),
        safety=SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=()),
        command_acks=(),
        recent_events=events,
        held_object_handle=held_object_handle,
    )


def test_snapshot_to_dict_has_required_keys():
    snap = _make_snapshot()
    d = snapshot_to_dict(snap)

    required = {
        "snapshot_id",
        "ts_ms",
        "arm_id",
        "skill",
        "target",
        "held_object_handle",
        "perception",
        "act",
        "progress",
        "outcome",
        "safety",
        "command_acks",
        "recent_events",
    }
    assert required <= d.keys()


def test_snapshot_to_dict_null_skill_and_target():
    snap = _make_snapshot(skill=None, target=None)
    d = snapshot_to_dict(snap)

    assert d["skill"] is None
    assert d["target"] is None
    assert d["held_object_handle"] is None


def test_snapshot_to_dict_with_active_skill():
    skill = SkillInfo(
        name=SkillName.PICK,
        skill_run_id="run-42",
        phase=PhaseId.SELECT_GRASP,
    )
    snap = _make_snapshot(skill=skill)
    d = snapshot_to_dict(snap)

    assert d["skill"] == {
        "name": "PICK",
        "skill_run_id": "run-42",
        "phase": "SELECT_GRASP",
    }


def test_snapshot_to_dict_with_held_object():
    snap = _make_snapshot(held_object_handle="cube-red-1")
    d = snapshot_to_dict(snap)
    assert d["held_object_handle"] == "cube-red-1"


def test_snapshot_to_dict_recent_events_included():
    event = EventEnvelope(
        event_id="ev-1",
        type=EventType.SKILL_SUCCEEDED,
        ts_ms=500,
        arm_id=ARM,
        data={"run_id": "run-1"},
    )
    snap = _make_snapshot(events=(event,))
    d = snapshot_to_dict(snap)

    assert len(d["recent_events"]) == 1
    ev_dict = d["recent_events"][0]
    assert ev_dict["event_id"] == "ev-1"
    assert ev_dict["type"] == "SKILL_SUCCEEDED"
    assert ev_dict["data"] == {"run_id": "run-1"}


def test_start_skill_tool_rejects_invalid_skill_name():
    ctx = AgentContext(arm_id=ARM, snapshot_id="snap-1")
    tools = build_tools(ctx)
    start_tool = next(t for t in tools if t.name == "start_skill")

    result = start_tool.invoke({"skill_name": "pick", "target_handle": "obj-1"})

    assert "REJECTED" in result
    assert ctx.commands == []
