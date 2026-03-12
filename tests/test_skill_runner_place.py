"""Tests for PLACE skill via SkillRunnerService."""

import pytest

from halo.contracts.actions import JointPositionChunk
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
RUN_ID = "run-place-1"


def _cfg(**kwargs) -> SkillRunnerConfig:
    kwargs.setdefault("runner_rate_hz", 10.0)
    return SkillRunnerConfig(**kwargs)


def _happy_cfg() -> SkillRunnerConfig:
    """Zero all timing so each phase fires in one tick."""
    return _cfg(
        select_place_timeout_ms=99999,
        transit_preplace_timeout_ms=99999,
        descend_place_timeout_ms=99999,
        place_align_threshold_m=0.10,
        place_distance_threshold_m=0.02,
        open_gripper_duration_ms=0,
        retreat_duration_ms=0,
        returning_timeout_ms=0,
        no_target_tolerance_ms=99999,
        recover_wait_ms=0,
        max_reacquire_attempts=3,
    )


def _target(distance_m: float = 0.5, hint_valid: bool = True, handle: str = "ref-1") -> TargetInfo:
    return TargetInfo(
        handle=handle,
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


async def _null_chunk_fn(arm_id: str, phase: PhaseId, snap: object) -> None:
    return None


def _make_svc(
    rt: HALORuntime,
    cfg: SkillRunnerConfig | None = None,
) -> SkillRunnerService:
    async def push_fn(chunk: JointPositionChunk) -> None:
        pass

    return SkillRunnerService(
        arm_id=ARM,
        runtime=rt,
        chunk_fn=_null_chunk_fn,
        push_fn=push_fn,
        config=cfg or _cfg(),
    )


@pytest.fixture
def rt() -> HALORuntime:
    r = HALORuntime()
    r.register_arm(ARM)
    return r


async def _seed_store(
    rt: HALORuntime, distance_m: float = 0.5, handle: str = "ref-1", held_object: str | None = "obj-held"
) -> None:
    await rt.store.update_target(ARM, _target(distance_m=distance_m, handle=handle))
    await rt.store.update_perception(ARM, _perception())
    await rt.store.update_act(ARM, _act(fill_ms=0))
    await rt.store.update_held_object_handle(ARM, held_object)
    await rt.get_latest_runtime_snapshot(ARM)


# --- Happy path ---


async def test_place_happy_path(rt: HALORuntime):
    """PLACE runs through SELECT_PLACE → TRANSIT_PREPLACE → DESCEND_PLACE → OPEN → RETREAT → DONE."""
    svc = _make_svc(rt, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1")

    # SELECT_PLACE → TRANSIT_PREPLACE (tracking OK, handle matches)
    await _seed_store(rt, distance_m=0.5)
    result = await svc.tick()
    assert result == PhaseId.TRANSIT_PREPLACE

    # TRANSIT_PREPLACE → DESCEND_PLACE (distance < align threshold)
    await _seed_store(rt, distance_m=0.05)
    result = await svc.tick()
    assert result == PhaseId.DESCEND_PLACE

    # DESCEND_PLACE → OPEN (distance < place threshold)
    await _seed_store(rt, distance_m=0.01)
    result = await svc.tick()
    assert result == PhaseId.OPEN

    # OPEN → RETREAT (timer elapsed, duration=0)
    result = await svc.tick()
    assert result == PhaseId.RETREAT

    # RETREAT → RETURNING (timer elapsed, duration=0)
    result = await svc.tick()
    assert result == PhaseId.RETURNING

    # RETURNING → DONE (timer elapsed, duration=0)
    result = await svc.tick()
    assert result == PhaseId.DONE

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_SUCCEEDED for e in events)


async def test_place_success_clears_held_object_handle(rt: HALORuntime):
    """After PLACE succeeds, held_object_handle is cleared to None."""
    svc = _make_svc(rt, cfg=_happy_cfg())

    # Pre-set held_object_handle as if PICK just succeeded
    await rt.store.update_held_object_handle(ARM, "obj-1")
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.held_object_handle == "obj-1"

    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1")

    # Drive through all phases
    await _seed_store(rt, distance_m=0.5)
    await svc.tick()  # SELECT_PLACE → TRANSIT_PREPLACE
    await _seed_store(rt, distance_m=0.05)
    await svc.tick()  # → DESCEND_PLACE
    await _seed_store(rt, distance_m=0.01)
    await svc.tick()  # → OPEN
    await svc.tick()  # → RETREAT
    await svc.tick()  # → RETURNING
    await svc.tick()  # → DONE

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.held_object_handle is None


async def test_place_options_stored_in_state_bag(rt: HALORuntime):
    """Options passed to start_skill are stored in state_bag."""
    svc = _make_svc(rt)
    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1", options={"modifier": "PLACE_NEXT_TO"})

    assert svc._active_run is not None
    assert svc._active_run.state_bag.get("modifier") == "PLACE_NEXT_TO"


# --- Guard: no held object ---


async def test_place_fails_without_held_object(rt: HALORuntime):
    """PLACE fails immediately with PLACE_MISS if no object is held."""
    svc = _make_svc(rt, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1")

    await _seed_store(rt, distance_m=0.5, held_object=None)
    result = await svc.tick()
    assert result == PhaseId.DONE

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events)
    failed = next(e for e in events if e.type == EventType.SKILL_FAILED)
    assert failed.data.get("failure_code") == SkillFailureCode.PLACE_MISS.value


# --- Timeouts and failures ---


async def test_place_select_place_timeout(rt: HALORuntime):
    """SELECT_PLACE times out when tracking not established."""
    cfg = _cfg(select_place_timeout_ms=0)
    svc = _make_svc(rt, cfg=cfg)
    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1")
    await _seed_store(rt, distance_m=0.5)

    await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    assert any(e.type == EventType.SKILL_FAILED for e in events)
    failed = next(e for e in events if e.type == EventType.SKILL_FAILED)
    assert failed.data.get("failure_code") == SkillFailureCode.PERCEPTION_LOST.value


async def test_place_descend_timeout_place_miss(rt: HALORuntime):
    """DESCEND_PLACE timeout results in PLACE_MISS failure."""
    cfg = _cfg(
        select_place_timeout_ms=99999,
        transit_preplace_timeout_ms=99999,
        descend_place_timeout_ms=0,
        place_align_threshold_m=0.10,
        place_distance_threshold_m=0.02,
        no_target_tolerance_ms=99999,
    )
    svc = _make_svc(rt, cfg=cfg)
    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1")

    # SELECT_PLACE → TRANSIT_PREPLACE
    await _seed_store(rt, distance_m=0.5)
    await svc.tick()

    # TRANSIT_PREPLACE → DESCEND_PLACE
    await _seed_store(rt, distance_m=0.05)
    await svc.tick()

    # DESCEND_PLACE timeout (distance still too far, timeout=0)
    await _seed_store(rt, distance_m=0.05)
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.outcome.state == SkillOutcomeState.FAILURE
    assert snap.outcome.reason_code == SkillFailureCode.PLACE_MISS


# --- Recovery ---


async def test_place_recovery_on_target_loss(rt: HALORuntime):
    """Target loss during TRANSIT_PREPLACE triggers RECOVER_RETRY_APPROACH."""
    cfg = _cfg(
        select_place_timeout_ms=99999,
        transit_preplace_timeout_ms=99999,
        no_target_tolerance_ms=0,
        place_align_threshold_m=0.10,
        recover_wait_ms=0,
        max_reacquire_attempts=3,
    )
    svc = _make_svc(rt, cfg=cfg)
    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1")

    # SELECT_PLACE → TRANSIT_PREPLACE
    await _seed_store(rt, distance_m=0.5)
    await svc.tick()
    assert svc._fsm.phase == PhaseId.TRANSIT_PREPLACE

    # Lose target — first tick sets no_target_start_ms
    await rt.store.update_target(ARM, _target(hint_valid=False))
    await rt.get_latest_runtime_snapshot(ARM)
    await svc.tick()
    # Second tick — tolerance exceeded (no_target_tolerance_ms=0)
    await svc.tick()
    assert svc._fsm.phase == PhaseId.RECOVER_RETRY_APPROACH

    # Recovery → back to TRANSIT_PREPLACE (recover_wait_ms=0)
    await svc.tick()
    assert svc._fsm.phase == PhaseId.TRANSIT_PREPLACE


# --- Events ---


async def test_place_emits_correct_events(rt: HALORuntime):
    """PLACE emits SKILL_STARTED, PHASE_ENTER/EXIT, SKILL_SUCCEEDED."""
    q = rt.bus.subscribe(ARM, maxsize=0)
    svc = _make_svc(rt, cfg=_happy_cfg())
    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1")

    await _seed_store(rt, distance_m=0.5)
    await svc.tick()  # SELECT_PLACE → TRANSIT_PREPLACE
    await _seed_store(rt, distance_m=0.05)
    await svc.tick()  # → DESCEND_PLACE
    await _seed_store(rt, distance_m=0.01)
    await svc.tick()  # → OPEN
    await svc.tick()  # → RETREAT
    await svc.tick()  # → RETURNING
    await svc.tick()  # → DONE

    all_events = []
    while not q.empty():
        all_events.append(q.get_nowait())
    types = [e.type for e in all_events]
    assert EventType.SKILL_STARTED in types
    assert EventType.SKILL_SUCCEEDED in types
    assert types.count(EventType.PHASE_ENTER) >= 7  # initial + 6 transitions
    assert types.count(EventType.PHASE_EXIT) >= 6
    rt.bus.unsubscribe(ARM, q)


async def test_place_needs_verify_false(rt: HALORuntime):
    """PLACE sets needs_verify=False in outcome."""
    svc = _make_svc(rt)
    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1")

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.outcome.needs_verify is False


async def test_place_initial_phase_is_select_place(rt: HALORuntime):
    """PLACE starts in SELECT_PLACE phase."""
    svc = _make_svc(rt)
    await svc.start_skill(SkillName.PLACE, RUN_ID, "ref-1")

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.skill is not None
    assert snap.skill.phase == PhaseId.SELECT_PLACE
    assert snap.skill.name == SkillName.PLACE
