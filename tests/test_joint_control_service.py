"""Tests for JointPositionControlService: FIFO buffer, safety, ActInfo tracking."""

import pytest

from halo.contracts.actions import JointPositionAction, JointPositionChunk
from halo.contracts.enums import ActStatus, PhaseId, SafetyState
from halo.contracts.events import EventType
from halo.runtime.runtime import HALORuntime
from halo.services.control_service.config import JointControlConfig
from halo.services.control_service.joint_service import JointPositionControlService

ARM = "arm0"


def _cfg(**kwargs) -> JointControlConfig:
    return JointControlConfig(**kwargs)


def _action(v0: float = 0.0) -> JointPositionAction:
    return JointPositionAction(values=(v0, 0.0, 0.0, 0.0, 0.0, 0.5))


def _chunk(n: int = 3, v0: float = 0.0) -> JointPositionChunk:
    return JointPositionChunk(
        chunk_id="test-chunk",
        arm_id=ARM,
        phase_id=PhaseId.MOVE_PREGRASP,
        actions=tuple(_action(v0) for _ in range(n)),
        ts_ms=0,
    )


@pytest.fixture
def rt() -> HALORuntime:
    r = HALORuntime()
    r.register_arm(ARM)
    return r


def _make_svc(rt: HALORuntime, cfg: JointControlConfig | None = None) -> JointPositionControlService:
    return JointPositionControlService(arm_id=ARM, runtime=rt, config=cfg or _cfg())


# --- push + pop FIFO ---


async def test_push_and_pop_fifo(rt: HALORuntime):
    svc = _make_svc(rt)
    await svc.push_chunk(_chunk(n=3))
    assert len(svc._buffer) == 3

    action = await svc.tick()
    assert action is not None
    assert len(svc._buffer) == 2


async def test_empty_buffer_returns_none_and_idle(rt: HALORuntime):
    svc = _make_svc(rt)
    action = await svc.tick()
    assert action is None

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.IDLE


# --- joint-limit safety check ---


async def test_out_of_range_triggers_reflex(rt: HALORuntime):
    svc = _make_svc(rt)
    # Push an action with shoulder_pan = 3.0 (out of range ±1.92)
    bad_chunk = JointPositionChunk(
        chunk_id="bad",
        arm_id=ARM,
        phase_id=PhaseId.MOVE_PREGRASP,
        actions=(JointPositionAction(values=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0)),),
        ts_ms=0,
    )
    await svc.push_chunk(bad_chunk)
    action = await svc.tick()
    assert action is None  # Held due to safety violation

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.safety.state == SafetyState.FAULT
    assert snap.safety.reflex_active is True


async def test_reflex_recovery_after_clean_action(rt: HALORuntime):
    svc = _make_svc(rt)
    # Bad action → reflex
    bad_chunk = JointPositionChunk(
        chunk_id="bad",
        arm_id=ARM,
        phase_id=PhaseId.MOVE_PREGRASP,
        actions=(JointPositionAction(values=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0)),),
        ts_ms=0,
    )
    await svc.push_chunk(bad_chunk)
    await svc.tick()
    assert svc._reflex_active is True

    # Clean action → recover
    await svc.push_chunk(_chunk(n=1))
    action = await svc.tick()
    assert action is not None
    assert svc._reflex_active is False

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.safety.state == SafetyState.OK


# --- ActInfo tracking ---


async def test_act_info_running_with_buffer(rt: HALORuntime):
    svc = _make_svc(rt, cfg=_cfg(buffer_low_threshold_ms=50))
    await svc.push_chunk(_chunk(n=5))
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.RUNNING
    assert snap.act.buffer_fill_ms > 0


# --- buffer clear on PHASE_ENTER ---


async def test_buffer_clears_on_phase_enter(rt: HALORuntime):
    svc = _make_svc(rt)
    await svc.push_chunk(_chunk(n=5))
    assert len(svc._buffer) == 5

    # Simulate PHASE_ENTER event
    from halo.contracts.events import EventEnvelope

    event = EventEnvelope(
        event_id="test",
        type=EventType.PHASE_ENTER,
        ts_ms=0,
        arm_id=ARM,
        data={"phase_id": int(PhaseId.EXECUTE_APPROACH)},
    )
    await svc._on_phase_event(event)
    assert len(svc._buffer) == 0
