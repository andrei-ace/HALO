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
from halo.services.skill_runner_service.definitions import SkillRegistry
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
    registry: SkillRegistry | None = None,
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
        registry=registry,
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
    empty_registry = SkillRegistry()
    svc, _ = _make_svc(rt, registry=empty_registry)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill is not None
    assert snap.skill.phase == PhaseId.DONE
    assert snap.outcome.state == SkillOutcomeState.FAILURE
    assert snap.outcome.reason_code == SkillFailureCode.PLANNER_ABORT

    events = rt.bus.get_recent_events(ARM)
    failed = [e for e in events if e.type == EventType.SKILL_FAILED]
    assert len(failed) == 1


async def test_start_skill_rejects_unsupported_skill_without_clobbering_active(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    # Use an empty registry svc to attempt a second skill — but since the
    # rejection path doesn't check registry when active run exists, we test
    # by using default registry + unknown variant instead.
    await svc.start_skill(SkillName.PICK, "run-bad", "obj-1", variant="nonexistent")

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill is not None
    assert snap.skill.skill_run_id == RUN_ID
    assert snap.skill.phase == PhaseId.SELECT_GRASP
    assert snap.outcome.state == SkillOutcomeState.IN_PROGRESS

    events = rt.bus.get_recent_events(ARM)
    failed = [e for e in events if e.type == EventType.SKILL_FAILED]
    assert any(e.data.get("skill_run_id") == "run-bad" for e in failed)


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
    assert failed.data.get("failure_code") == SkillFailureCode.PERCEPTION_LOST.value


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


async def test_pick_success_sets_held_object_handle(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

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
    assert snap.held_object_handle == "obj-1"


async def test_outcome_info_updated_in_store_on_failure(rt: HALORuntime):
    cfg = _cfg(select_grasp_timeout_ms=0, no_target_tolerance_ms=99999)
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=cfg)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.5)

    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.outcome.state == SkillOutcomeState.FAILURE
    assert snap.outcome.reason_code == SkillFailureCode.PERCEPTION_LOST
    assert snap.held_object_handle is None


async def test_progress_info_updated_in_store(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.5)

    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.progress.elapsed_ms >= 0


# --- Sim mode ---


def _make_sim_phase_fn(phase_sequence: list[tuple[int, bool]], *, error: str | None = None):
    """Create a sim_phase_fn that plays through a fixed phase sequence."""
    idx = 0

    def sim_phase_fn() -> tuple[int, bool, str | None]:
        nonlocal idx
        entry = phase_sequence[min(idx, len(phase_sequence) - 1)]
        phase_id, done = entry[0], entry[1]
        # Allow per-entry error via 3-tuples, or use the default error kwarg
        err = entry[2] if len(entry) > 2 else (error if done else None)
        idx += 1
        return phase_id, done, err

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
    await _seed_store(rt, distance_m=0.5)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    # Tick through all sim phases (+1 for SELECT_GRASP→PLAN_APPROACH via advance)
    for _ in range(7):
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
    await _seed_store(rt, distance_m=0.5)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    await svc.tick()  # SELECT_GRASP→PLAN_APPROACH (advance, tracking OK) + triggers sim pick
    await svc.tick()  # MOVE_PREGRASP (sync from sim)
    await svc.tick()  # EXECUTE_APPROACH (sync from sim)

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
    await _seed_store(rt, distance_m=0.5)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await svc.tick()  # SELECT_GRASP→PLAN_APPROACH + trigger sim pick
    await svc.tick()  # sync MOVE_PREGRASP from sim

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
    await _seed_store(rt, distance_m=0.5)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    # First tick: SELECT_GRASP→PLAN_APPROACH, triggers start_pick_fn which returns error
    await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events)


async def test_sim_mode_deferred_planning_failure(rt: HALORuntime):
    """Sim mode: done=True with early phase_id (deferred GraspPlanningFailure) emits SKILL_FAILED."""
    # Simulate what happens when execute_pending_pick fails:
    # server accepts start_pick (done=False), then planning fails (done=True, phase stays at 0).
    svc = _make_sim_svc(
        rt,
        phase_sequence=[
            (int(PhaseId.IDLE), False),  # server accepted, trajectory pending
            (int(PhaseId.IDLE), True),  # planning failed, done=True
        ],
    )
    await _seed_store(rt, distance_m=0.5)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    # Tick 1: SELECT_GRASP→PLAN_APPROACH (advance, tracking OK) + triggers sim pick
    await svc.tick()
    # Tick 2: sim_phase_fn returns (0, False) — server accepted, waiting
    await svc.tick()
    # Tick 3: sim_phase_fn returns (0, True) — deferred planning failure
    await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events)
    assert not any(e.type == EventType.SKILL_SUCCEEDED for e in events)


async def test_sim_mode_verify_grasp_error_fails_skill(rt: HALORuntime):
    """Sim mode: NO_GRASP error from VERIFY_GRASP propagates as SKILL_FAILED.

    Reproduces the bug where the sim server detects a failed grasp (Z delta
    below threshold) but the error is cleared by the return trajectory,
    causing the skill to report SUCCESS instead of FAILURE.
    """
    # Simulate: normal phases → return trajectory completes with NO_GRASP error
    svc = _make_sim_svc(
        rt,
        phase_sequence=[
            (int(PhaseId.MOVE_PREGRASP), False),
            (int(PhaseId.EXECUTE_APPROACH), False),
            (int(PhaseId.CLOSE_GRIPPER), False),
            (int(PhaseId.LIFT), False),
            (int(PhaseId.VERIFY_GRASP), False),
            # Return trajectory completes with error preserved
            (60, True, "NO_GRASP"),  # PHASE_RETURNING=60, done=True, error
        ],
    )
    await _seed_store(rt, distance_m=0.5)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    for _ in range(10):
        await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events), "Expected SKILL_FAILED for NO_GRASP"
    assert not any(e.type == EventType.SKILL_SUCCEEDED for e in events), "Should NOT succeed with NO_GRASP error"


# --- Sim mode PLACE tests ---


def _make_mock_start_place_fn(success: bool = True):
    async def start_place_fn(arm_id: str, target_body: str, held_body: str) -> dict:
        if success:
            return {"type": "start_place_ok", "target_body": target_body}
        return {"type": "start_place_error", "message": "mock failure"}

    return start_place_fn


def _make_sim_place_svc(
    rt: HALORuntime,
    phase_sequence: list[tuple[int, bool]] | None = None,
    start_place_success: bool = True,
) -> SkillRunnerService:
    if phase_sequence is None:
        # Server trajectory reports the last segment phase with done=True,
        # not PhaseId.DONE(9). HALO interprets done=True as DONE.
        phase_sequence = [
            (int(PhaseId.TRANSIT_PREPLACE), False),
            (int(PhaseId.DESCEND_PLACE), False),
            (int(PhaseId.OPEN), False),
            (int(PhaseId.RETREAT), False),
            (int(PhaseId.RETREAT), True),
        ]

    svc = SkillRunnerService(
        arm_id=ARM,
        runtime=rt,
        config=_cfg(),
        start_pick_fn=_make_mock_start_pick_fn(),
        start_place_fn=_make_mock_start_place_fn(start_place_success),
        sim_phase_fn=_make_sim_phase_fn(phase_sequence),
    )
    return svc


async def test_sim_place_happy_path(rt: HALORuntime):
    """Sim PLACE: SELECT_PLACE → trigger → sync through TRANSIT_PREPLACE..DONE."""
    svc = _make_sim_place_svc(rt)
    await _seed_store(rt, distance_m=0.5)
    await rt.store.update_held_object_handle(ARM, "obj-1")
    await svc.start_skill(SkillName.PLACE, RUN_ID, "obj-1")

    # Tick 1: SELECT_PLACE → TRANSIT_PREPLACE (handler advance) + triggers sim place
    await svc.tick()
    # Ticks 2-6: sync phases from sim telemetry
    for _ in range(5):
        await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_SUCCEEDED for e in events)

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.held_object_handle is None


async def test_sim_place_stale_telemetry_ignored(rt: HALORuntime):
    """Sim PLACE: stale done=True from previous PICK is ignored until fresh telemetry arrives."""
    phase_sequence = [
        # Stale frames from previous PICK (done=True, phase_id=LIFT)
        (int(PhaseId.LIFT), True),
        (int(PhaseId.LIFT), True),
        # Fresh frames: server acknowledged start_place
        (int(PhaseId.TRANSIT_PREPLACE), False),
        (int(PhaseId.DESCEND_PLACE), False),
        (int(PhaseId.RETREAT), False),
        (int(PhaseId.RETREAT), True),
    ]
    svc = _make_sim_place_svc(rt, phase_sequence=phase_sequence)
    await _seed_store(rt, distance_m=0.5)
    await rt.store.update_held_object_handle(ARM, "obj-1")
    await svc.start_skill(SkillName.PLACE, RUN_ID, "obj-1")

    # Tick 1: gate (SELECT_PLACE → TRANSIT_PREPLACE) + trigger
    await svc.tick()
    # Ticks 2-7: stale frames skipped, then fresh frames synced
    for _ in range(6):
        await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_SUCCEEDED for e in events)


async def test_sim_place_immediate_ik_failure(rt: HALORuntime):
    """Sim PLACE: all done=True frames (IK failed immediately) → SKILL_FAILED after stale guard timeout."""
    # All frames are (IDLE, True) — simulates immediate IK failure with no done=False frame ever sent
    phase_sequence = [
        (int(PhaseId.IDLE), True),
        (int(PhaseId.IDLE), True),
        (int(PhaseId.IDLE), True),
        (int(PhaseId.IDLE), True),
        (int(PhaseId.IDLE), True),
    ]
    svc = SkillRunnerService(
        arm_id=ARM,
        runtime=rt,
        config=_cfg(sim_stale_guard_timeout_ms=100),
        start_pick_fn=_make_mock_start_pick_fn(),
        start_place_fn=_make_mock_start_place_fn(),
        sim_phase_fn=_make_sim_phase_fn(phase_sequence),
    )
    await _seed_store(rt, distance_m=0.5)
    await rt.store.update_held_object_handle(ARM, "obj-1")
    await svc.start_skill(SkillName.PLACE, RUN_ID, "obj-1")

    # Tick 1: SELECT_PLACE → TRANSIT_PREPLACE + trigger
    await svc.tick()

    # Wait for stale guard timeout to expire
    import asyncio

    await asyncio.sleep(0.15)

    # Tick 2+: stale guard times out, accepts done=True → SKILL_FAILED
    for _ in range(3):
        await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events)
    assert not any(e.type == EventType.SKILL_SUCCEEDED for e in events)


async def test_sim_pick_immediate_ik_failure(rt: HALORuntime):
    """Sim PICK: all done=True frames (IK failed immediately) → SKILL_FAILED after stale guard timeout."""
    phase_sequence = [
        (int(PhaseId.IDLE), True),
        (int(PhaseId.IDLE), True),
        (int(PhaseId.IDLE), True),
        (int(PhaseId.IDLE), True),
        (int(PhaseId.IDLE), True),
    ]
    svc = SkillRunnerService(
        arm_id=ARM,
        runtime=rt,
        config=_cfg(sim_stale_guard_timeout_ms=100),
        start_pick_fn=_make_mock_start_pick_fn(),
        sim_phase_fn=_make_sim_phase_fn(phase_sequence),
    )
    await _seed_store(rt, distance_m=0.5)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    # Tick 1: SELECT_GRASP → PLAN_APPROACH + trigger
    await svc.tick()

    import asyncio

    await asyncio.sleep(0.15)

    # Tick 2+: stale guard times out, accepts done=True → SKILL_FAILED
    for _ in range(3):
        await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events)
    assert not any(e.type == EventType.SKILL_SUCCEEDED for e in events)
