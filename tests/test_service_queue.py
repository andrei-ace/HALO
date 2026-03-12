"""Tests for skill queue integration with SkillRunnerService."""

import pytest

from halo.contracts.enums import (
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


def _happy_cfg() -> SkillRunnerConfig:
    return SkillRunnerConfig(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        grasp_distance_threshold_m=0.02,
        grasp_persistence_ms=0,
        select_grasp_timeout_ms=99999,
        plan_approach_timeout_ms=99999,
        close_gripper_duration_ms=0,
        verify_duration_ms=0,
        lift_duration_ms=0,
        skip_verify_grasp=True,
        no_target_tolerance_ms=99999,
        move_pregrasp_timeout_ms=99999,
        visual_align_timeout_ms=99999,
        execute_approach_timeout_ms=99999,
    )


@pytest.fixture
async def rt() -> HALORuntime:
    runtime = HALORuntime()
    runtime.register_arm(ARM)
    return runtime


def _make_svc(runtime: HALORuntime, cfg: SkillRunnerConfig | None = None) -> SkillRunnerService:
    async def noop_chunk(arm_id, phase, snap):
        return None

    async def noop_push(chunk):
        pass

    return SkillRunnerService(
        arm_id=ARM,
        runtime=runtime,
        chunk_fn=noop_chunk,
        push_fn=noop_push,
        config=cfg or _happy_cfg(),
    )


async def _seed(rt: HALORuntime, distance_m: float = 0.5):
    await seed_store(
        rt,
        ARM,
        target=make_target(handle="obj-1", distance_m=distance_m, hint_valid=True),
        perception=make_perception(tracking_status=TrackingStatus.TRACKING),
    )
    await rt.get_latest_runtime_snapshot(ARM)


async def _drive_to_done(svc: SkillRunnerService, rt: HALORuntime):
    """Drive the active pick skill through all phases to DONE."""
    await _seed(rt, 0.5)
    await svc.tick()  # SELECT_GRASP -> PLAN_APPROACH
    await svc.tick()  # PLAN_APPROACH -> MOVE_PREGRASP
    await _seed(rt, 0.10)
    await svc.tick()  # -> VISUAL_ALIGN
    await _seed(rt, 0.03)
    await svc.tick()  # -> EXECUTE_APPROACH
    await _seed(rt, 0.01)
    await svc.tick()  # -> CLOSE_GRIPPER
    await svc.tick()  # -> LIFT
    await svc.tick()  # -> DONE


async def test_enqueue_when_active(rt: HALORuntime):
    svc = _make_svc(rt)
    await _seed(rt)
    await svc.start_skill(SkillName.PICK, "run-1", "obj-1")

    # Second skill should be enqueued
    await svc.start_skill(SkillName.TRACK, "run-2", "obj-2")
    assert svc._queue.size == 1
    assert svc._queue.peek().skill_name == SkillName.TRACK


async def test_auto_advance_from_queue(rt: HALORuntime):
    svc = _make_svc(rt)
    await _seed(rt)
    await svc.start_skill(SkillName.PICK, "run-1", "obj-1")
    await svc.start_skill(SkillName.TRACK, "run-2", "obj-1")

    # Drive first skill to completion
    await _drive_to_done(svc, rt)

    # Queue should have auto-activated the TRACK skill
    assert svc._active_run is not None
    assert svc._active_run.skill_name == SkillName.TRACK
    assert svc._active_run.skill_run_id == "run-2"
    assert svc._queue.size == 0


async def test_abort_activates_next(rt: HALORuntime):
    svc = _make_svc(rt)
    await _seed(rt)
    await svc.start_skill(SkillName.PICK, "run-1", "obj-1")
    await svc.start_skill(SkillName.TRACK, "run-2", "obj-1")

    await svc.abort_skill()

    # Aborted skill should trigger next from queue
    assert svc._active_run is not None
    assert svc._active_run.skill_name == SkillName.TRACK


async def test_failure_clears_queue(rt: HALORuntime):
    cfg = SkillRunnerConfig(select_grasp_timeout_ms=0)
    svc = _make_svc(rt, cfg)
    await _seed(rt)
    await svc.start_skill(SkillName.PICK, "run-1", "obj-1")
    await svc.start_skill(SkillName.TRACK, "run-2", "obj-1")

    # First tick: timeout -> failure -> queue cleared (stale follow-ups invalid)
    await svc.tick()

    assert svc._queue.size == 0
    assert svc._active_run is not None
    assert svc._active_run.skill_name == SkillName.PICK  # completed (failed), not replaced


async def test_empty_queue_no_activation(rt: HALORuntime):
    svc = _make_svc(rt)
    await _seed(rt)
    await svc.start_skill(SkillName.PICK, "run-1", "obj-1")
    await _drive_to_done(svc, rt)

    # No queued skills — active_run should be the completed one
    assert svc._active_run.outcome == SkillOutcomeState.SUCCESS
    assert svc._queue.size == 0


async def test_queue_overflow_emits_failure(rt: HALORuntime):
    """When queue is full, start_skill emits SKILL_FAILED."""
    cfg = SkillRunnerConfig(max_queue_size=1)
    svc = _make_svc(rt, cfg)
    await _seed(rt)
    q = rt.bus.subscribe(ARM)

    await svc.start_skill(SkillName.PICK, "run-1", "obj-1")  # active
    await svc.start_skill(SkillName.TRACK, "run-2", "obj-2")  # queued (fills queue)
    await svc.start_skill(SkillName.PICK, "run-3", "obj-3")  # overflow

    assert svc._queue.size == 1  # only run-2

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    overflow_fails = [e for e in events if e.type == EventType.SKILL_FAILED and e.data.get("skill_run_id") == "run-3"]
    assert len(overflow_fails) == 1
    assert overflow_fails[0].data["reason"] == "queue_full"

    rt.bus.unsubscribe(ARM, q)


async def test_terminal_tick_does_not_schedule_chunk_for_queued(rt: HALORuntime):
    """After terminal transition auto-activates next skill, the tick must not
    schedule a chunk using stale snapshot data from the previous run."""
    chunks_pushed: list = []

    async def tracking_chunk(arm_id, phase, snap):
        # Should never be called for TRACK skill
        from halo.contracts.actions import ZERO_JOINT_ACTION, JointPositionChunk

        chunks_pushed.append(phase)
        actions = (ZERO_JOINT_ACTION,) * 10
        return JointPositionChunk(chunk_id="c", arm_id=arm_id, phase_id=phase, actions=actions, ts_ms=0)

    async def noop_push(chunk):
        pass

    cfg = _happy_cfg()
    svc = SkillRunnerService(
        arm_id=ARM,
        runtime=rt,
        chunk_fn=tracking_chunk,
        push_fn=noop_push,
        config=cfg,
    )

    await _seed(rt, 0.5)
    await svc.start_skill(SkillName.PICK, "run-1", "obj-1")
    await svc.start_skill(SkillName.TRACK, "run-2", "obj-1")

    # Drive to the tick that completes the PICK skill
    await _seed(rt, 0.5)
    await svc.tick()  # SELECT_GRASP -> PLAN_APPROACH
    await svc.tick()  # PLAN_APPROACH -> MOVE_PREGRASP
    await _seed(rt, 0.10)
    await svc.tick()  # -> VISUAL_ALIGN
    await _seed(rt, 0.03)
    await svc.tick()  # -> EXECUTE_APPROACH
    await _seed(rt, 0.01)
    await svc.tick()  # -> CLOSE_GRIPPER
    await svc.tick()  # -> LIFT
    chunks_pushed.clear()
    await svc.tick()  # -> DONE + auto-activate TRACK

    # The TRACK skill is now active but the tick that triggered DONE should
    # NOT have called chunk_fn for the newly activated TRACK run.
    from halo.contracts.enums import PhaseId

    track_chunks = [p for p in chunks_pushed if p == PhaseId.ACQUIRING]
    assert len(track_chunks) == 0


async def test_view_model_with_queue(rt: HALORuntime):
    svc = _make_svc(rt)
    await _seed(rt)
    await svc.start_skill(SkillName.PICK, "run-1", "obj-1")
    await svc.start_skill(SkillName.TRACK, "run-2", "obj-2")

    vm = svc.get_view_model()
    assert vm is not None
    assert vm.skill_name == "PICK"
    assert len(vm.queued_skills) == 1
    assert vm.queued_skills[0].skill_name == "TRACK"
