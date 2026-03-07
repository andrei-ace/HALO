"""Service tests for SkillRunnerService with TRACK skill."""

import asyncio

import pytest

from halo.contracts.enums import (
    PhaseId,
    SkillFailureCode,
    SkillName,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.events import EventType
from halo.runtime.runtime import HALORuntime
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.service import SkillRunnerService
from halo.testing.state_seeder import make_perception, make_target, seed_store

ARM = "arm0"


@pytest.fixture
async def rt() -> HALORuntime:
    runtime = HALORuntime()
    runtime.register_arm(ARM)
    return runtime


def _make_svc(runtime: HALORuntime, **kw) -> SkillRunnerService:
    async def noop_chunk(arm_id, phase, snap):
        return None

    async def noop_push(chunk):
        pass

    return SkillRunnerService(
        arm_id=ARM,
        runtime=runtime,
        chunk_fn=kw.get("chunk_fn", noop_chunk),
        push_fn=kw.get("push_fn", noop_push),
        config=kw.get("config", SkillRunnerConfig(select_grasp_timeout_ms=5000)),
    )


async def test_track_creates_track_fsm(rt: HALORuntime):
    """Starting a TRACK skill activates ACQUIRING phase."""
    svc = _make_svc(rt)
    await svc.start_skill(SkillName.TRACK, "run-1", "cube-1")
    assert svc._active_run is not None
    assert svc._active_run.skill_name == SkillName.TRACK
    assert svc._fsm.phase == PhaseId.ACQUIRING


async def test_track_succeeds_when_tracking_established(rt: HALORuntime):
    """TRACK skill succeeds when perception reports TRACKING for the target."""
    svc = _make_svc(rt)
    q = rt.bus.subscribe(ARM)

    # Seed store with tracking target
    await seed_store(
        rt,
        ARM,
        target=make_target(handle="cube-1", hint_valid=True),
        perception=make_perception(tracking_status=TrackingStatus.TRACKING),
    )

    await svc.start_skill(SkillName.TRACK, "run-1", "cube-1")
    await svc.tick()

    # Should have succeeded
    assert svc._fsm.phase == PhaseId.DONE
    assert svc._fsm.outcome == SkillOutcomeState.SUCCESS

    # Check events
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    succeeded = [e for e in events if e.type == EventType.SKILL_SUCCEEDED]
    assert len(succeeded) == 1

    rt.bus.unsubscribe(ARM, q)


async def test_track_fails_on_timeout(rt: HALORuntime):
    """TRACK skill fails with PERCEPTION_LOST on timeout."""
    cfg = SkillRunnerConfig(acquiring_timeout_ms=0, acquiring_retry_budget=1)  # instant timeout
    svc = _make_svc(rt, config=cfg)
    q = rt.bus.subscribe(ARM)

    await seed_store(
        rt,
        ARM,
        target=None,
        perception=make_perception(tracking_status=TrackingStatus.IDLE),
    )

    await svc.start_skill(SkillName.TRACK, "run-1", "cube-1")
    await svc.tick()

    assert svc._fsm.phase == PhaseId.DONE
    assert svc._fsm.outcome == SkillOutcomeState.FAILURE
    assert svc._fsm.failure_code == SkillFailureCode.PERCEPTION_LOST

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    failed = [e for e in events if e.type == EventType.SKILL_FAILED]
    assert len(failed) == 1

    rt.bus.unsubscribe(ARM, q)


async def test_track_waits_for_retry_budget_before_failing(rt: HALORuntime):
    """TRACK remains active until the full acquisition retry budget is consumed."""
    cfg = SkillRunnerConfig(acquiring_timeout_ms=100, acquiring_retry_budget=3)
    svc = _make_svc(rt, config=cfg)

    await seed_store(
        rt,
        ARM,
        target=None,
        perception=make_perception(tracking_status=TrackingStatus.REACQUIRING),
    )

    await svc.start_skill(SkillName.TRACK, "run-1", "cube-1")
    await svc.tick()
    assert svc._fsm.phase == PhaseId.ACQUIRING
    assert svc._fsm.outcome == SkillOutcomeState.IN_PROGRESS

    await seed_store(
        rt,
        ARM,
        target=None,
        perception=make_perception(tracking_status=TrackingStatus.REACQUIRING),
    )
    await asyncio.sleep(0.31)
    await svc.tick()

    assert svc._fsm.phase == PhaseId.DONE
    assert svc._fsm.outcome == SkillOutcomeState.FAILURE
    assert svc._fsm.failure_code == SkillFailureCode.PERCEPTION_LOST


async def test_track_does_not_update_held_object_handle(rt: HALORuntime):
    """TRACK success must NOT set held_object_handle."""
    svc = _make_svc(rt)

    await seed_store(
        rt,
        ARM,
        target=make_target(handle="cube-1", hint_valid=True),
        perception=make_perception(tracking_status=TrackingStatus.TRACKING),
    )

    await svc.start_skill(SkillName.TRACK, "run-1", "cube-1")
    await svc.tick()

    assert svc._fsm.outcome == SkillOutcomeState.SUCCESS
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.held_object_handle is None


async def test_track_emits_correct_events(rt: HALORuntime):
    """TRACK emits SKILL_STARTED, PHASE_ENTER, PHASE_EXIT, SKILL_SUCCEEDED."""
    svc = _make_svc(rt)
    q = rt.bus.subscribe(ARM)

    await seed_store(
        rt,
        ARM,
        target=make_target(handle="cube-1", hint_valid=True),
        perception=make_perception(tracking_status=TrackingStatus.TRACKING),
    )

    await svc.start_skill(SkillName.TRACK, "run-1", "cube-1")
    await svc.tick()

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    types = [e.type for e in events]
    assert EventType.SKILL_STARTED in types
    assert EventType.SKILL_SUCCEEDED in types
    assert EventType.PHASE_ENTER in types

    # SKILL_STARTED should have skill_name=TRACK
    started = [e for e in events if e.type == EventType.SKILL_STARTED]
    assert started[0].data["skill_name"] == SkillName.TRACK

    rt.bus.unsubscribe(ARM, q)
