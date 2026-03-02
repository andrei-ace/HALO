"""Tests for SkillRunnerService: tick, start/stop, chunk scheduling, events."""

import pytest

from halo.contracts.actions import ZERO_ACTION, ActionChunk
from halo.contracts.enums import (
    ActStatus,
    PerceptionFailureCode,
    PhaseId,
    SkillFailureCode,
    SkillName,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.events import EventType
from halo.contracts.snapshots import ActInfo, PerceptionInfo, TargetInfo
from halo.runtime.runtime import HALORuntime
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.service import SkillRunnerService

ARM = "arm0"
RUN_ID = "run-test-1"


def _cfg(**kwargs) -> SkillRunnerConfig:
    kwargs.setdefault("runner_rate_hz", 10.0)
    return SkillRunnerConfig(**kwargs)


def _happy_cfg() -> SkillRunnerConfig:
    """Zero all timing so each phase fires in one tick."""
    return _cfg(
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


def _target(distance_m: float = 0.5, hint_valid: bool = True) -> TargetInfo:
    return TargetInfo(
        handle="obj-1",
        hint_valid=hint_valid,
        confidence=0.9,
        obs_age_ms=10,
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, distance_m),
        distance_m=distance_m,
    )


def _perception() -> PerceptionInfo:
    return PerceptionInfo(
        tracking_status=TrackingStatus.TRACKING,
        failure_code=PerceptionFailureCode.OK,
        reacquire_fail_count=0,
        vlm_job_pending=False,
    )


def _act(fill_ms: int = 0) -> ActInfo:
    return ActInfo(
        status=ActStatus.RUNNING if fill_ms > 0 else ActStatus.IDLE,
        buffer_fill_ms=fill_ms,
        buffer_low=fill_ms == 0,
    )


def _make_chunk(phase: PhaseId) -> ActionChunk:
    actions = tuple(ZERO_ACTION for _ in range(10))
    return ActionChunk(
        chunk_id="chunk-test",
        arm_id=ARM,
        phase_id=phase,
        actions=actions,
        ts_ms=0,
    )


async def _null_chunk_fn(arm_id: str, phase: PhaseId, snap: object) -> None:
    return None


async def _fixed_chunk_fn(arm_id: str, phase: PhaseId, snap: object) -> ActionChunk:
    return _make_chunk(phase)


@pytest.fixture
def rt() -> HALORuntime:
    r = HALORuntime()
    r.register_arm(ARM)
    return r


def _make_svc(
    rt: HALORuntime,
    chunk_fn=None,
    push_fn=None,
    cfg: SkillRunnerConfig | None = None,
) -> tuple[SkillRunnerService, list[ActionChunk]]:
    chunks_pushed: list[ActionChunk] = []

    if chunk_fn is None:
        chunk_fn = _fixed_chunk_fn

    if push_fn is None:

        async def push_fn(chunk: ActionChunk) -> None:
            chunks_pushed.append(chunk)

    svc = SkillRunnerService(
        arm_id=ARM,
        runtime=rt,
        chunk_fn=chunk_fn,
        push_fn=push_fn,
        config=cfg or _cfg(),
    )
    return svc, chunks_pushed


async def _seed_store(rt: HALORuntime, distance_m: float = 0.5) -> None:
    await rt.store.update_target(ARM, _target(distance_m=distance_m))
    await rt.store.update_perception(ARM, _perception())
    await rt.store.update_act(ARM, _act(fill_ms=0))
    # Build snapshot so get_latest_snapshot() returns non-None
    await rt.get_latest_runtime_snapshot(ARM)


# --- start_skill publishes events ---


async def test_start_skill_publishes_skill_started_and_phase_enter(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    events = rt.bus.get_recent_events(ARM)
    types = [e.type for e in events]
    assert EventType.SKILL_STARTED in types
    assert EventType.PHASE_ENTER in types

    phase_enter = next(e for e in events if e.type == EventType.PHASE_ENTER)
    assert phase_enter.data["phase_id"] == int(PhaseId.SELECT_GRASP)


async def test_start_skill_updates_store_skill_info(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill is not None
    assert snap.skill.phase == PhaseId.SELECT_GRASP
    assert snap.skill.skill_run_id == RUN_ID


async def test_start_skill_rejects_unsupported_skill(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    await svc.start_skill(SkillName.PLACE, RUN_ID, "obj-1")

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill is not None
    assert snap.skill.phase == PhaseId.DONE
    assert snap.outcome.state == SkillOutcomeState.FAILURE
    assert snap.outcome.reason_code == SkillFailureCode.UNSAFE_ABORT

    events = rt.bus.get_recent_events(ARM)
    failed = [e for e in events if e.type == EventType.SKILL_FAILED]
    assert len(failed) == 1


async def test_start_skill_rejects_unsupported_skill_without_clobbering_active(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await svc.start_skill(SkillName.PLACE, "run-place", "obj-1")

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill is not None
    assert snap.skill.skill_run_id == RUN_ID
    assert snap.skill.phase == PhaseId.SELECT_GRASP
    assert snap.outcome.state == SkillOutcomeState.IN_PROGRESS

    events = rt.bus.get_recent_events(ARM)
    failed = [e for e in events if e.type == EventType.SKILL_FAILED]
    assert any(e.data.get("skill_run_id") == "run-place" for e in failed)


# --- tick before start ---


async def test_tick_before_start_skill_returns_none(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    result = await svc.tick()
    assert result is None


# --- tick advances phases ---


async def test_tick_advances_through_initial_phases(rt: HALORuntime):
    """SELECT_GRASP and PLAN_APPROACH are v0 pass-throughs; first tick goes to MOVE_PREGRASP."""
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.5)

    # First tick: SELECT_GRASP -> PLAN_APPROACH (immediate pass-through)
    result = await svc.tick()
    assert result == PhaseId.PLAN_APPROACH


async def test_tick_publishes_phase_exit_and_phase_enter_on_transition(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.5)

    await svc.tick()  # SELECT_GRASP -> PLAN_APPROACH

    events = rt.bus.get_recent_events(ARM)
    types = [e.type for e in events]
    assert EventType.PHASE_EXIT in types
    assert EventType.PHASE_ENTER in types

    exit_evt = next(e for e in events if e.type == EventType.PHASE_EXIT)
    assert exit_evt.data["phase_id"] == int(PhaseId.SELECT_GRASP)


async def test_tick_updates_store_skill_info_on_transition(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.5)

    await svc.tick()  # SELECT_GRASP -> PLAN_APPROACH

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill.phase == PhaseId.PLAN_APPROACH


# --- chunk scheduling ---


async def test_tick_pushes_chunk_when_buffer_low(rt: HALORuntime):
    svc, chunks = _make_svc(rt, cfg=_cfg(buffer_target_ms=200))
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    # Seed with fill_ms=0 -> needs_chunk=True
    await rt.store.update_target(ARM, _target(distance_m=0.5))
    await rt.store.update_perception(ARM, _perception())
    await rt.store.update_act(ARM, _act(fill_ms=0))
    await rt.get_latest_runtime_snapshot(ARM)

    await svc.tick()
    assert len(chunks) == 1


async def test_tick_does_not_push_chunk_when_buffer_full(rt: HALORuntime):
    svc, chunks = _make_svc(rt, cfg=_cfg(buffer_target_ms=200))
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    # Seed with fill_ms=300 -> needs_chunk=False
    await rt.store.update_target(ARM, _target(distance_m=0.5))
    await rt.store.update_perception(ARM, _perception())
    await rt.store.update_act(ARM, _act(fill_ms=300))
    await rt.get_latest_runtime_snapshot(ARM)

    await svc.tick()
    assert len(chunks) == 0


# --- happy-path full run ---


async def test_tick_publishes_skill_succeeded_on_happy_path(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    # Drive through SELECT_GRASP -> PLAN_APPROACH -> MOVE_PREGRASP
    await _seed_store(rt, distance_m=0.5)
    await svc.tick()  # SELECT_GRASP -> PLAN_APPROACH
    await svc.tick()  # PLAN_APPROACH -> MOVE_PREGRASP

    # MOVE_PREGRASP -> VISUAL_ALIGN
    await _seed_store(rt, distance_m=0.10)
    await svc.tick()

    # VISUAL_ALIGN -> EXECUTE_APPROACH
    await _seed_store(rt, distance_m=0.03)
    await svc.tick()

    # EXECUTE_APPROACH -> CLOSE_GRIPPER (persistence=0 -> immediate)
    await _seed_store(rt, distance_m=0.01)
    await svc.tick()

    # CLOSE_GRIPPER -> LIFT (duration=0, skip_verify=True)
    await svc.tick()

    # LIFT -> DONE (duration=0)
    await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_SUCCEEDED for e in events)


# --- timeout failure ---


async def test_tick_publishes_skill_failed_on_timeout(rt: HALORuntime):
    cfg = _cfg(select_grasp_timeout_ms=0, no_target_tolerance_ms=99999)
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=cfg)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.5)

    await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events)

    failed = next(e for e in events if e.type == EventType.SKILL_FAILED)
    assert failed.data.get("failure_code") == SkillFailureCode.NO_PROGRESS.value


# --- abort_skill ---


async def test_abort_skill_publishes_skill_failed(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    await svc.abort_skill()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events)
    assert any(e.type == EventType.PHASE_EXIT for e in events)


# --- start/stop lifecycle ---


async def test_start_stop_lifecycle(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    await svc.start()
    assert svc._loop_task is not None

    await svc.stop()
    assert svc._loop_task is None


# --- store outcome/progress updates ---


async def test_outcome_info_updated_in_store_on_success(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    # Drive through all phases
    await _seed_store(rt, distance_m=0.5)
    await svc.tick()  # SELECT_GRASP -> PLAN_APPROACH
    await svc.tick()  # PLAN_APPROACH -> MOVE_PREGRASP
    await _seed_store(rt, distance_m=0.10)
    await svc.tick()  # -> VISUAL_ALIGN
    await _seed_store(rt, distance_m=0.03)
    await svc.tick()  # -> EXECUTE_APPROACH
    await _seed_store(rt, distance_m=0.01)
    await svc.tick()  # -> CLOSE_GRIPPER
    await svc.tick()  # -> LIFT
    await svc.tick()  # -> DONE

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.outcome.state == SkillOutcomeState.SUCCESS
    assert snap.outcome.reason_code is None


async def test_outcome_info_updated_in_store_on_failure(rt: HALORuntime):
    cfg = _cfg(select_grasp_timeout_ms=0, no_target_tolerance_ms=99999)
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=cfg)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.5)

    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.outcome.state == SkillOutcomeState.FAILURE
    assert snap.outcome.reason_code == SkillFailureCode.NO_PROGRESS


async def test_progress_info_updated_in_store(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.5)

    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.progress.elapsed_ms >= 0


# --- Sim mode ---


def _make_sim_phase_fn(phase_sequence: list[tuple[int, bool]]):
    """Create a sim_phase_fn that plays through a fixed phase sequence."""
    idx = 0

    def sim_phase_fn() -> tuple[int, bool]:
        nonlocal idx
        phase_id, done = phase_sequence[min(idx, len(phase_sequence) - 1)]
        idx += 1
        return phase_id, done

    return sim_phase_fn


def _make_mock_start_pick_fn(success: bool = True):
    """Create a start_pick_fn that immediately returns success or error."""

    async def start_pick_fn(arm_id: str, target_body: str) -> dict:
        if success:
            return {"type": "start_pick_ok", "duration": 3.0}
        return {"type": "start_pick_error", "message": "mock failure"}

    return start_pick_fn


def _make_sim_svc(
    rt: HALORuntime,
    phase_sequence: list[tuple[int, bool]] | None = None,
    start_pick_success: bool = True,
) -> SkillRunnerService:
    if phase_sequence is None:
        phase_sequence = [
            (int(PhaseId.MOVE_PREGRASP), False),
            (int(PhaseId.EXECUTE_APPROACH), False),
            (int(PhaseId.CLOSE_GRIPPER), False),
            (int(PhaseId.LIFT), False),
            (int(PhaseId.DONE), True),
        ]

    svc = SkillRunnerService(
        arm_id=ARM,
        runtime=rt,
        config=_cfg(),
        start_pick_fn=_make_mock_start_pick_fn(start_pick_success),
        sim_phase_fn=_make_sim_phase_fn(phase_sequence),
    )
    return svc


async def test_sim_mode_phase_transitions(rt: HALORuntime):
    """Sim mode syncs phases from sim_phase_fn."""
    svc = _make_sim_svc(rt)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    # Tick through all sim phases
    for _ in range(5):
        await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_SUCCEEDED for e in events)


async def test_sim_mode_publishes_phase_events(rt: HALORuntime):
    """Sim mode publishes PHASE_EXIT/PHASE_ENTER on transitions."""
    svc = _make_sim_svc(
        rt,
        phase_sequence=[
            (int(PhaseId.MOVE_PREGRASP), False),
            (int(PhaseId.EXECUTE_APPROACH), False),
        ],
    )
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    await svc.tick()  # MOVE_PREGRASP (transition from SELECT_GRASP)
    await svc.tick()  # EXECUTE_APPROACH (transition from MOVE_PREGRASP)

    events = rt.bus.get_recent_events(ARM)
    exits = [e for e in events if e.type == EventType.PHASE_EXIT]
    enters = [e for e in events if e.type == EventType.PHASE_ENTER]
    assert len(exits) >= 2
    assert len(enters) >= 3  # SELECT_GRASP enter + 2 transitions


async def test_sim_mode_updates_act_info(rt: HALORuntime):
    """Sim mode updates ActInfo in store."""
    svc = _make_sim_svc(
        rt,
        phase_sequence=[(int(PhaseId.MOVE_PREGRASP), False)],
    )
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.act.status == ActStatus.RUNNING


async def test_sim_mode_rejects_both_modes(rt: HALORuntime):
    """Cannot provide both ACT and sim callables."""
    with pytest.raises(ValueError, match="Cannot provide both"):
        SkillRunnerService(
            arm_id=ARM,
            runtime=rt,
            chunk_fn=_null_chunk_fn,
            push_fn=_null_chunk_fn,
            start_pick_fn=_make_mock_start_pick_fn(),
        )


async def test_sim_mode_start_pick_error_fails_skill(rt: HALORuntime):
    """Sim mode: start_pick_fn returning error emits SKILL_FAILED."""
    svc = _make_sim_svc(rt, start_pick_success=False)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    # start_skill should have failed due to start_pick_error
    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events)
