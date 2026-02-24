"""Tests for ControlService: tick, push_chunk, start/stop, reflex, phase trim."""

import asyncio

import pytest

from halo.contracts.actions import ZERO_ACTION, Action, ActionChunk
from halo.contracts.enums import ActStatus, PhaseId, SafetyState
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import TargetInfo
from halo.runtime.runtime import HALORuntime
from halo.services.control_service.config import ControlServiceConfig
from halo.services.control_service.service import ControlService

ARM = "arm0"
RATE = 50.0


def _cfg(**kwargs) -> ControlServiceConfig:
    kwargs.setdefault("control_rate_hz", RATE)
    return ControlServiceConfig(**kwargs)


def _safe_chunk(
    n: int,
    arm_id: str = ARM,
    phase: PhaseId = PhaseId.APPROACH_PREGRASP,
) -> ActionChunk:
    """Chunk with actions well within safety limits (dx=0.001)."""
    actions = tuple(Action(0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(n))
    return ActionChunk(chunk_id="c", arm_id=arm_id, phase_id=phase, actions=actions, ts_ms=0)


def _bad_chunk(arm_id: str = ARM) -> ActionChunk:
    """Chunk with a single over-limit action (dx=0.1 >> 0.01 limit)."""
    actions = (Action(dx=0.1, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, gripper_cmd=0.0),)
    return ActionChunk(chunk_id="bad", arm_id=arm_id, phase_id=PhaseId.APPROACH_PREGRASP, actions=actions, ts_ms=0)


@pytest.fixture
def rt() -> HALORuntime:
    r = HALORuntime()
    r.register_arm(ARM)
    return r


@pytest.fixture
def applied() -> list[tuple[str, Action]]:
    return []


@pytest.fixture
def svc(rt: HALORuntime, applied: list) -> ControlService:
    async def apply_fn(arm_id: str, action: Action) -> None:
        applied.append((arm_id, action))

    return ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg())


# --- tick on empty buffer ---


async def test_tick_empty_buffer_returns_none(svc: ControlService, rt: HALORuntime):
    result = await svc.tick()
    assert result is None
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.IDLE


async def test_tick_empty_buffer_does_not_call_apply_fn(svc: ControlService, applied: list):
    await svc.tick()
    assert len(applied) == 0


# --- tick after push_chunk ---


async def test_tick_after_push_applies_action(svc: ControlService, rt: HALORuntime, applied: list):
    await svc.push_chunk(_safe_chunk(10))
    result = await svc.tick()
    assert result is not None
    assert len(applied) == 1
    assert applied[0][0] == ARM
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.RUNNING


async def test_tick_returns_clamped_action(svc: ControlService, applied: list):
    await svc.push_chunk(_safe_chunk(5))
    result = await svc.tick()
    assert result is not None
    # sub-limit action should pass through unchanged
    assert result.dx == pytest.approx(0.001)


# --- buffer drops below threshold ---


async def test_tick_buffer_low_status(rt: HALORuntime):
    actions_applied: list[Action] = []

    async def apply_fn(arm_id: str, action: Action) -> None:
        actions_applied.append(action)

    # Default threshold is 100 ms → 5 actions at 50 Hz
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg())
    # Push 4 actions (80 ms) — already below threshold after one pop
    await svc.push_chunk(_safe_chunk(4))
    await svc.tick()  # pops 1, leaves 3 (60 ms) → buffer_low=True
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.buffer_low is True
    assert snap.act.status == ActStatus.BUFFER_LOW


# --- safety violation → reflex ---


async def test_tick_safety_violation_triggers_reflex(rt: HALORuntime):
    actions_applied: list[Action] = []

    async def apply_fn(arm_id: str, action: Action) -> None:
        actions_applied.append(action)

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg())
    await svc.push_chunk(_bad_chunk())
    result = await svc.tick()

    assert result is None
    # ZERO_ACTION applied as hold
    assert len(actions_applied) == 1
    assert actions_applied[0] == ZERO_ACTION
    # Reflex state set
    assert svc._reflex_active is True
    # Event published
    recent = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SAFETY_REFLEX_TRIGGERED for e in recent)
    # Store updated
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.safety.state == SafetyState.FAULT
    assert snap.safety.reflex_active is True


async def test_tick_safety_violation_does_not_re_trigger_reflex(rt: HALORuntime):
    """Second tick with same violation should not publish a second SAFETY_REFLEX_TRIGGERED."""
    events_published: list[EventEnvelope] = []  # noqa: F841

    async def apply_fn(arm_id: str, action: Action) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg())
    # Two bad actions
    await svc.push_chunk(_bad_chunk())
    await svc.push_chunk(_bad_chunk())
    await svc.tick()  # triggers reflex
    reflex_count_after_first = sum(
        1 for e in rt.bus.get_recent_events(ARM) if e.type == EventType.SAFETY_REFLEX_TRIGGERED
    )
    await svc.tick()  # reflex already active — should not publish again
    reflex_count_after_second = sum(
        1 for e in rt.bus.get_recent_events(ARM) if e.type == EventType.SAFETY_REFLEX_TRIGGERED
    )
    assert reflex_count_after_first == reflex_count_after_second


# --- stale hint → hold, no reflex ---


async def test_tick_stale_hint_applies_zero_action(rt: HALORuntime):
    actions_applied: list[Action] = []

    async def apply_fn(arm_id: str, action: Action) -> None:
        actions_applied.append(action)

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg(max_obs_age_ms=200))
    stale_target = TargetInfo(
        handle="obj-1",
        hint_valid=False,
        confidence=0.0,
        obs_age_ms=500,
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, 0.0),
        distance_m=0.5,
    )
    await rt.store.update_target(ARM, stale_target)
    await svc.push_chunk(_safe_chunk(5))

    result = await svc.tick()

    assert result is None
    assert len(actions_applied) == 1
    assert actions_applied[0] == ZERO_ACTION
    # No safety reflex should be triggered
    recent = rt.bus.get_recent_events(ARM)
    assert not any(e.type == EventType.SAFETY_REFLEX_TRIGGERED for e in recent)
    # Act status is STALE
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.STALE


async def test_tick_stale_hint_does_not_set_reflex_active(rt: HALORuntime):
    async def apply_fn(arm_id: str, action: Action) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg())
    stale_target = TargetInfo(
        handle="obj-1",
        hint_valid=True,
        confidence=0.9,
        obs_age_ms=999,  # way above 200 ms threshold
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, 0.0),
        distance_m=0.5,
    )
    await rt.store.update_target(ARM, stale_target)
    await svc.push_chunk(_safe_chunk(5))
    await svc.tick()
    assert svc._reflex_active is False


# --- push_chunk from wrong arm ---


async def test_push_chunk_wrong_arm_raises(svc: ControlService):
    bad_chunk = _safe_chunk(3, arm_id="arm1")
    with pytest.raises(ValueError, match="arm_id"):
        await svc.push_chunk(bad_chunk)


# --- phase PHASE_ENTER event trims buffer ---


async def test_phase_enter_event_trims_buffer(rt: HALORuntime):
    async def apply_fn(arm_id: str, action: Action) -> None:
        pass

    cfg = _cfg(buffer_trim_ms=75)  # int(75 * 50 / 1000) = 3 actions
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=cfg)
    await svc.push_chunk(_safe_chunk(20))  # 400 ms

    event = EventEnvelope(
        event_id="evt-phase",
        type=EventType.PHASE_ENTER,
        ts_ms=0,
        arm_id=ARM,
        data={"phase_id": int(PhaseId.ALIGN)},
    )
    await svc._on_phase_event(event)

    assert svc._buffer.size <= 3


async def test_non_phase_enter_event_does_not_trim(rt: HALORuntime):
    async def apply_fn(arm_id: str, action: Action) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg())
    await svc.push_chunk(_safe_chunk(20))

    event = EventEnvelope(
        event_id="evt-other",
        type=EventType.PHASE_EXIT,
        ts_ms=0,
        arm_id=ARM,
        data={},
    )
    await svc._on_phase_event(event)

    assert svc._buffer.size == 20


# --- start / stop lifecycle ---


async def test_start_stop_cleans_up_tasks(rt: HALORuntime):
    async def apply_fn(arm_id: str, action: Action) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg())
    await svc.start()
    assert svc._loop_task is not None
    assert svc._event_task is not None

    await svc.stop()
    assert svc._loop_task is None
    assert svc._event_task is None
    assert svc._phase_queue is None


async def test_start_stop_loop_runs_ticks(rt: HALORuntime):
    tick_count = 0

    async def apply_fn(arm_id: str, action: Action) -> None:
        nonlocal tick_count
        tick_count += 1

    cfg = _cfg(control_rate_hz=200.0)  # fast loop for test
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=cfg)
    await svc.push_chunk(_safe_chunk(100))
    await svc.start()
    await asyncio.sleep(0.05)  # ~10 ticks at 200 Hz
    await svc.stop()

    assert tick_count >= 1


# --- reflex clears on clean tick ---


async def test_reflex_clears_on_clean_tick(rt: HALORuntime):
    async def apply_fn(arm_id: str, action: Action) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg())

    # Trigger reflex with a bad action
    await svc.push_chunk(_bad_chunk())
    await svc.tick()
    assert svc._reflex_active is True

    # Push a safe action → reflex should clear
    await svc.push_chunk(_safe_chunk(1))
    result = await svc.tick()

    assert result is not None
    assert svc._reflex_active is False
    # SAFETY_RECOVERED event published
    recent = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SAFETY_RECOVERED for e in recent)
    # Store updated
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.safety.state == SafetyState.OK
    assert snap.safety.reflex_active is False


async def test_reflex_does_not_clear_while_violations_persist(rt: HALORuntime):
    async def apply_fn(arm_id: str, action: Action) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, config=_cfg())

    # Two bad actions
    await svc.push_chunk(_bad_chunk())
    await svc.push_chunk(_bad_chunk())
    await svc.tick()  # triggers reflex
    await svc.tick()  # still bad → reflex stays active

    assert svc._reflex_active is True
    recent = rt.bus.get_recent_events(ARM)
    assert not any(e.type == EventType.SAFETY_RECOVERED for e in recent)
