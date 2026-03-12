"""Tests for ControlService: tick, push_chunk, start/stop, reflex, phase trim."""

import asyncio

import pytest

from halo.bridge import BridgeTransportError
from halo.contracts.actions import ZERO_JOINT_ACTION, JointPositionAction, JointPositionChunk
from halo.contracts.enums import ActStatus, PhaseId, SafetyState, SkillName
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import SkillInfo, TargetInfo
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
    phase: PhaseId = PhaseId.MOVE_PREGRASP,
) -> JointPositionChunk:
    """Chunk with actions well within safety limits."""
    actions = tuple(JointPositionAction(values=(0.001, 0.0, 0.0, 0.0, 0.0, 0.5)) for _ in range(n))
    return JointPositionChunk(chunk_id="c", arm_id=arm_id, phase_id=phase, actions=actions, ts_ms=0)


def _bad_chunk(arm_id: str = ARM) -> JointPositionChunk:
    """Chunk with a single out-of-range action (shoulder_pan=3.0 >> ±1.92 limit)."""
    actions = (JointPositionAction(values=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0)),)
    return JointPositionChunk(chunk_id="bad", arm_id=arm_id, phase_id=PhaseId.MOVE_PREGRASP, actions=actions, ts_ms=0)


@pytest.fixture
def rt() -> HALORuntime:
    r = HALORuntime()
    r.register_arm(ARM)
    return r


@pytest.fixture
def applied() -> list[tuple[str, JointPositionAction]]:
    return []


@pytest.fixture
def svc(rt: HALORuntime, applied: list) -> ControlService:
    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        applied.append((arm_id, action))

    return ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())


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
    assert result.values[0] == pytest.approx(0.001)


async def test_tick_tracks_last_applied(svc: ControlService):
    """After a successful tick, _last_applied is updated to the applied action."""
    assert svc._last_applied == ZERO_JOINT_ACTION
    await svc.push_chunk(_safe_chunk(5))
    result = await svc.tick()
    assert result is not None
    assert svc._last_applied == result


# --- buffer drops below threshold ---


async def test_tick_buffer_low_status(rt: HALORuntime):
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    # Default threshold is 100 ms → 5 actions at 50 Hz
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    # Push 4 actions (80 ms) — already below threshold after one pop
    await svc.push_chunk(_safe_chunk(4))
    await svc.tick()  # pops 1, leaves 3 (60 ms) → buffer_low=True
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.buffer_low is True
    assert snap.act.status == ActStatus.BUFFER_LOW


# --- safety violation → reflex ---


async def test_tick_safety_violation_triggers_reflex(rt: HALORuntime):
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    await svc.push_chunk(_bad_chunk())
    result = await svc.tick()

    assert result is None
    # apply_fn called with ZERO_JOINT_ACTION (initial hold target)
    assert len(actions_applied) == 1
    assert all(v == 0.0 for v in actions_applied[0].values)
    # Reflex state set
    assert svc._reflex_active is True
    # Event published
    recent = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SAFETY_REFLEX_TRIGGERED for e in recent)
    # Store updated
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.safety.state == SafetyState.FAULT
    assert snap.safety.reflex_active is True


async def test_tick_safety_violation_holds_last_position(rt: HALORuntime):
    """When _last_applied exists, reflex re-applies it (hold position)."""
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    # First: apply a safe action so _last_applied is set
    await svc.push_chunk(_safe_chunk(1))
    await svc.tick()
    assert len(actions_applied) == 1
    held_position = actions_applied[0]

    # Now push a bad action
    await svc.push_chunk(_bad_chunk())
    await svc.tick()

    # Should re-apply the last safe position, not zeros
    assert len(actions_applied) == 2
    assert actions_applied[1] == held_position


async def test_tick_safety_violation_does_not_re_trigger_reflex(rt: HALORuntime):
    """Second tick with same violation should not publish a second SAFETY_REFLEX_TRIGGERED."""
    events_published: list[EventEnvelope] = []  # noqa: F841

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
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


async def test_tick_stale_hint_holds_last_position(rt: HALORuntime):
    """Stale hint re-applies _last_applied (hold position), not zeros."""
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    cfg = _cfg(max_obs_age_ms=200)
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=cfg)

    # Apply a safe action first
    await svc.push_chunk(_safe_chunk(1))
    await svc.tick()
    held_position = actions_applied[0]

    # Now set stale target
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
    assert len(actions_applied) == 2
    assert actions_applied[1] == held_position  # re-applied last position, not zeros
    # No safety reflex should be triggered
    recent = rt.bus.get_recent_events(ARM)
    assert not any(e.type == EventType.SAFETY_REFLEX_TRIGGERED for e in recent)
    # Act status is STALE
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.STALE


async def test_tick_stale_hint_applies_initial_state_hold(rt: HALORuntime):
    """Stale hint on first tick applies initial_state (ZERO_JOINT_ACTION) as hold."""
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    cfg = _cfg(max_obs_age_ms=200)
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=cfg)
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
    # Hold command sent using initial_state (ZERO_JOINT_ACTION)
    assert len(actions_applied) == 1
    assert all(v == 0.0 for v in actions_applied[0].values)
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.STALE


async def test_tick_stale_hint_does_not_set_reflex_active(rt: HALORuntime):
    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
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
    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    cfg = _cfg(buffer_trim_ms=75)  # int(75 * 50 / 1000) = 3 actions
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=cfg)
    await svc.push_chunk(_safe_chunk(20))  # 400 ms

    event = EventEnvelope(
        event_id="evt-phase",
        type=EventType.PHASE_ENTER,
        ts_ms=0,
        arm_id=ARM,
        data={"phase_id": int(PhaseId.VISUAL_ALIGN)},
    )
    await svc._on_phase_event(event)

    assert svc._buffer.size <= 3


async def test_non_phase_enter_event_does_not_trim(rt: HALORuntime):
    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
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
    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    await svc.start()
    assert svc._loop_task is not None
    assert svc._event_task is not None

    await svc.stop()
    assert svc._loop_task is None
    assert svc._event_task is None
    assert svc._phase_queue is None


async def test_start_stop_loop_runs_ticks(rt: HALORuntime):
    tick_count = 0

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        nonlocal tick_count
        tick_count += 1

    cfg = _cfg(control_rate_hz=200.0)  # fast loop for test
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=cfg)
    await svc.push_chunk(_safe_chunk(100))
    await svc.start()
    await asyncio.sleep(0.05)  # ~10 ticks at 200 Hz
    await svc.stop()

    assert tick_count >= 1


# --- reflex clears on clean tick ---


async def test_reflex_clears_on_clean_tick(rt: HALORuntime):
    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())

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
    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())

    # Two bad actions
    await svc.push_chunk(_bad_chunk())
    await svc.push_chunk(_bad_chunk())
    await svc.tick()  # triggers reflex
    await svc.tick()  # still bad → reflex stays active

    assert svc._reflex_active is True
    recent = rt.bus.get_recent_events(ARM)
    assert not any(e.type == EventType.SAFETY_RECOVERED for e in recent)


# --- wrist_enabled propagation ---


async def test_wrist_enabled_false_when_no_skill(svc: ControlService, rt: HALORuntime):
    """No skill active → wrist_enabled=False in ActInfo."""
    await svc.tick()
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.wrist_enabled is False


async def test_wrist_enabled_false_in_non_wrist_phase(rt: HALORuntime):
    """MOVE_PREGRASP is not a wrist-active phase → wrist_enabled=False."""

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    await rt.store.update_skill(ARM, SkillInfo(name=SkillName.PICK, skill_run_id="r1", phase=PhaseId.MOVE_PREGRASP))
    await svc.push_chunk(_safe_chunk(5))
    await svc.tick()
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.wrist_enabled is False


async def test_wrist_enabled_true_in_visual_align(rt: HALORuntime):
    """VISUAL_ALIGN is a wrist-active phase → wrist_enabled=True."""

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    await rt.store.update_skill(ARM, SkillInfo(name=SkillName.PICK, skill_run_id="r1", phase=PhaseId.VISUAL_ALIGN))
    await svc.push_chunk(_safe_chunk(5))
    await svc.tick()
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.wrist_enabled is True


async def test_wrist_enabled_true_in_execute_approach(rt: HALORuntime):
    """EXECUTE_APPROACH is a wrist-active phase → wrist_enabled=True."""

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    await rt.store.update_skill(ARM, SkillInfo(name=SkillName.PICK, skill_run_id="r1", phase=PhaseId.EXECUTE_APPROACH))
    await svc.push_chunk(_safe_chunk(5))
    await svc.tick()
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.wrist_enabled is True


async def test_wrist_enabled_propagates_on_stale_hint(rt: HALORuntime):
    """Stale-hint path should still write correct wrist_enabled."""

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        pass

    cfg = _cfg(max_obs_age_ms=200)
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=cfg)
    await rt.store.update_skill(ARM, SkillInfo(name=SkillName.PICK, skill_run_id="r1", phase=PhaseId.CLOSE_GRIPPER))
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
    await svc.tick()
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.STALE
    assert snap.act.wrist_enabled is True  # CLOSE_GRIPPER is wrist-active


# --- bridge transport failure ---


async def test_transport_failure_writes_stale_status(rt: HALORuntime):
    """apply_fn raising BridgeTransportError should result in STALE, not RUNNING."""

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        raise BridgeTransportError("timeout")

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    await svc.push_chunk(_safe_chunk(5))
    result = await svc.tick()

    assert result is None
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.STALE


async def test_transport_failure_returns_none(rt: HALORuntime):
    """Failed apply should return None (no action was applied)."""

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        raise BridgeTransportError("dead bridge")

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    await svc.push_chunk(_safe_chunk(5))
    result = await svc.tick()

    assert result is None


async def test_transport_failure_on_stale_hint_still_writes_stale(rt: HALORuntime):
    """Bridge failure during stale-hint hold should still write STALE (not crash)."""

    call_count = 0

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise BridgeTransportError("dead bridge")

    cfg = _cfg(max_obs_age_ms=200)
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=cfg)

    # Apply one safe action first to set _last_applied
    await svc.push_chunk(_safe_chunk(1))
    await svc.tick()

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
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.STALE


async def test_transport_failure_on_reflex_still_writes_status(rt: HALORuntime):
    """Bridge failure during safety reflex hold should not crash."""

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        raise BridgeTransportError("dead bridge")

    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=_cfg())
    await svc.push_chunk(_bad_chunk())
    result = await svc.tick()

    assert result is None
    # Reflex was still triggered despite bridge failure
    assert svc._reflex_active is True


# --- per-tick velocity limiting ---


async def test_tick_velocity_jump_is_clamped_not_reflexed(rt: HALORuntime):
    """Large joint jump is clamped to max_joint_delta_rad, not reflexed."""
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    cfg = _cfg(max_joint_delta_rad=0.1)
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=cfg)

    # First tick: apply position at 0.0
    chunk1 = JointPositionChunk(
        chunk_id="c1",
        arm_id=ARM,
        phase_id=PhaseId.MOVE_PREGRASP,
        actions=(JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),),
        ts_ms=0,
    )
    await svc.push_chunk(chunk1)
    await svc.tick()

    # Second tick: jump to 1.0 (delta=1.0 >> 0.1 limit) — clamped, not reflexed
    chunk2 = JointPositionChunk(
        chunk_id="c2",
        arm_id=ARM,
        phase_id=PhaseId.MOVE_PREGRASP,
        actions=(JointPositionAction(values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),),
        ts_ms=0,
    )
    await svc.push_chunk(chunk2)
    result = await svc.tick()

    assert result is not None
    assert result.values[0] == pytest.approx(0.1)  # clamped to 0.0 + 0.1
    assert svc._reflex_active is False


async def test_tick_velocity_clamp_limits_motion(rt: HALORuntime):
    """Clamp restricts per-tick motion to max_joint_delta_rad."""
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    cfg = _cfg(max_joint_delta_rad=0.1)
    initial = JointPositionAction(values=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=initial, config=cfg)

    # Request jump to 0.55 — small delta (0.05 < 0.1), should pass and not be clamped
    chunk2 = JointPositionChunk(
        chunk_id="c2",
        arm_id=ARM,
        phase_id=PhaseId.MOVE_PREGRASP,
        actions=(JointPositionAction(values=(0.55, 0.0, 0.0, 0.0, 0.0, 0.0)),),
        ts_ms=0,
    )
    await svc.push_chunk(chunk2)
    result = await svc.tick()

    assert result is not None
    assert result.values[0] == pytest.approx(0.55)


# --- initial_state seeding ---


async def test_initial_state_seeds_velocity_clamping(rt: HALORuntime):
    """initial_state seeds _last_applied so the first command is velocity-clamped."""
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    initial = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    cfg = _cfg(max_joint_delta_rad=0.1)
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=initial, config=cfg)

    # First command jumps to 1.0 — exceeds 0.1 limit from initial state at 0.0
    chunk = JointPositionChunk(
        chunk_id="c1",
        arm_id=ARM,
        phase_id=PhaseId.MOVE_PREGRASP,
        actions=(JointPositionAction(values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),),
        ts_ms=0,
    )
    await svc.push_chunk(chunk)
    result = await svc.tick()

    # Clamped to 0.1 from initial state, not reflexed
    assert result is not None
    assert result.values[0] == pytest.approx(0.1)
    assert svc._reflex_active is False


async def test_default_initial_state_velocity_clamps_first_tick(rt: HALORuntime):
    """Default initial_state (ZERO_JOINT_ACTION) clamps velocity from first tick."""
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    cfg = _cfg(max_joint_delta_rad=0.1)
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=ZERO_JOINT_ACTION, config=cfg)

    # Jump to 1.0 from default initial state (0.0) — delta 1.0 >> 0.1 limit
    chunk = JointPositionChunk(
        chunk_id="c1",
        arm_id=ARM,
        phase_id=PhaseId.MOVE_PREGRASP,
        actions=(JointPositionAction(values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),),
        ts_ms=0,
    )
    await svc.push_chunk(chunk)
    result = await svc.tick()

    # Clamped to 0.1, not reflexed
    assert result is not None
    assert result.values[0] == pytest.approx(0.1)
    assert svc._reflex_active is False


async def test_out_of_range_initial_state_is_clamped(rt: HALORuntime):
    """Out-of-range initial_state is clamped so hold commands stay within limits."""
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    # Shoulder_pan 2.5 exceeds ±1.92 limit
    out_of_range = JointPositionAction(values=(2.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    cfg = _cfg(max_obs_age_ms=200)
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=out_of_range, config=cfg)

    # _last_applied should be clamped at construction
    assert svc._last_applied.values[0] == pytest.approx(1.92)

    # Stale-hint tick re-applies _last_applied — must be in-range
    stale = TargetInfo(
        handle="x",
        hint_valid=False,
        confidence=0.5,
        obs_age_ms=999,
        time_skew_ms=0,
        delta_xyz_ee=(0, 0, 0),
        distance_m=0.5,
    )
    await rt.store.update_target(ARM, stale)
    await svc.tick()
    assert len(actions_applied) >= 1
    assert actions_applied[-1].values[0] == pytest.approx(1.92)


# --- gripper velocity exemption ---


async def test_gripper_large_jump_not_flagged(rt: HALORuntime):
    """Gripper full-range jump uses max_gripper_delta, not max_joint_delta_rad."""
    actions_applied: list[JointPositionAction] = []

    async def apply_fn(arm_id: str, action: JointPositionAction) -> None:
        actions_applied.append(action)

    cfg = _cfg(max_joint_delta_rad=0.1, max_gripper_delta=1.0)
    initial = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    svc = ControlService(arm_id=ARM, runtime=rt, apply_fn=apply_fn, initial_state=initial, config=cfg)

    # Gripper jumps 0.9 — exceeds arm limit (0.1) but within gripper limit (1.0)
    chunk = JointPositionChunk(
        chunk_id="c1",
        arm_id=ARM,
        phase_id=PhaseId.MOVE_PREGRASP,
        actions=(JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.9)),),
        ts_ms=0,
    )
    await svc.push_chunk(chunk)
    result = await svc.tick()

    assert result is not None
    assert result.values[5] == pytest.approx(0.9)
    assert svc._reflex_active is False
