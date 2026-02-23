"""Tests for SkillRunnerService: tick, start/stop, chunk scheduling, events."""

import pytest

from halo.contracts.actions import Action, ActionChunk, ZERO_ACTION
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
from halo.contracts.snapshots import ActInfo, OutcomeInfo, PerceptionInfo, ProgressInfo, TargetInfo
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
        descend_threshold_m=0.05,
        grasp_distance_threshold_m=0.02,
        grasp_persistence_ms=0,
        close_duration_ms=0,
        verify_duration_ms=0,
        lift_duration_ms=0,
        skip_verify_grasp=True,
        no_target_tolerance_ms=99999,
        approach_timeout_ms=99999,
        align_timeout_ms=99999,
        descend_timeout_ms=99999,
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
    assert phase_enter.data["phase_id"] == int(PhaseId.APPROACH_PREGRASP)


async def test_start_skill_updates_store_skill_info(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill is not None
    assert snap.skill.phase == PhaseId.APPROACH_PREGRASP
    assert snap.skill.skill_run_id == RUN_ID


# --- tick before start ---

async def test_tick_before_start_skill_returns_none(rt: HALORuntime):
    svc, _ = _make_svc(rt)
    result = await svc.tick()
    assert result is None


# --- tick advances phases ---

async def test_tick_advances_to_align_with_close_target(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.10)  # below approach_align_threshold=0.15

    result = await svc.tick()
    assert result == PhaseId.ALIGN


async def test_tick_publishes_phase_exit_and_phase_enter_on_transition(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.10)

    await svc.tick()  # APPROACH → ALIGN

    events = rt.bus.get_recent_events(ARM)
    types = [e.type for e in events]
    assert EventType.PHASE_EXIT in types
    assert EventType.PHASE_ENTER in types

    exit_evt = next(e for e in events if e.type == EventType.PHASE_EXIT)
    assert exit_evt.data["phase_id"] == int(PhaseId.APPROACH_PREGRASP)

    enter_evts = [e for e in events if e.type == EventType.PHASE_ENTER]
    assert any(e.data["phase_id"] == int(PhaseId.ALIGN) for e in enter_evts)


async def test_tick_updates_store_skill_info_on_transition(rt: HALORuntime):
    svc, _ = _make_svc(rt, chunk_fn=_null_chunk_fn, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    await _seed_store(rt, distance_m=0.10)

    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill.phase == PhaseId.ALIGN


# --- chunk scheduling ---

async def test_tick_pushes_chunk_when_buffer_low(rt: HALORuntime):
    svc, chunks = _make_svc(rt, cfg=_cfg(buffer_target_ms=200))
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    # Seed with fill_ms=0 → needs_chunk=True
    await rt.store.update_target(ARM, _target(distance_m=0.5))
    await rt.store.update_perception(ARM, _perception())
    await rt.store.update_act(ARM, _act(fill_ms=0))
    await rt.get_latest_runtime_snapshot(ARM)

    await svc.tick()
    assert len(chunks) == 1


async def test_tick_does_not_push_chunk_when_buffer_full(rt: HALORuntime):
    svc, chunks = _make_svc(rt, cfg=_cfg(buffer_target_ms=200))
    await svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
    # Seed with fill_ms=300 → needs_chunk=False
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

    # Drive through all phases
    await _seed_store(rt, distance_m=0.10)   # APPROACH→ALIGN
    await svc.tick()
    await _seed_store(rt, distance_m=0.03)   # ALIGN→DESCEND
    await svc.tick()
    await _seed_store(rt, distance_m=0.01)   # DESCEND→CLOSE (persistence=0 → immediate)
    await svc.tick()
    await svc.tick()                          # CLOSE→LIFT   (duration=0)
    await svc.tick()                          # LIFT→DONE    (duration=0)

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_SUCCEEDED for e in events)


# --- timeout failure ---

async def test_tick_publishes_skill_failed_on_timeout(rt: HALORuntime):
    cfg = _cfg(approach_timeout_ms=0, no_target_tolerance_ms=99999)
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

    await _seed_store(rt, distance_m=0.10)
    await svc.tick()
    await _seed_store(rt, distance_m=0.03)
    await svc.tick()
    await _seed_store(rt, distance_m=0.01)
    await svc.tick()
    await svc.tick()
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.outcome.state == SkillOutcomeState.SUCCESS
    assert snap.outcome.reason_code is None


async def test_outcome_info_updated_in_store_on_failure(rt: HALORuntime):
    cfg = _cfg(approach_timeout_ms=0, no_target_tolerance_ms=99999)
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
