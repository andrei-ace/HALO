"""Tests for FsmEngine."""

from halo.contracts.enums import (
    ActStatus,
    PerceptionFailureCode,
    PhaseId,
    SkillFailureCode,
    SkillName,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.snapshots import ActInfo, PerceptionInfo, TargetInfo
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.definitions import build_default_registry
from halo.services.skill_runner_service.engine import FsmEngine
from halo.services.skill_runner_service.skill_run import NodeStatus

T0 = 1000


def _cfg(**kw) -> SkillRunnerConfig:
    return SkillRunnerConfig(**kw)


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


def _act() -> ActInfo:
    return ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=300, buffer_low=False)


def _pick_engine(cfg: SkillRunnerConfig | None = None) -> FsmEngine:
    cfg = cfg or _cfg()
    registry = build_default_registry()
    defn = registry.get(SkillName.PICK)
    return FsmEngine(defn.graph, defn.handler_factory(), cfg, defn.global_guard_factory())


def _started_run(engine: FsmEngine):
    return engine.create_run(T0, "run-1", "obj-1")


def test_create_run_starts_at_entry():
    e = _pick_engine()
    run = _started_run(e)
    assert run.current_node == "SELECT_GRASP"
    assert run.phase_id == PhaseId.SELECT_GRASP
    assert run.is_active
    assert run.node_statuses["SELECT_GRASP"] == NodeStatus.ACTIVE


def test_advance_select_grasp_to_plan_approach():
    e = _pick_engine()
    run = _started_run(e)
    old = e.advance(run, T0 + 1, _target(), _perception(), _act())
    assert old == PhaseId.SELECT_GRASP
    assert run.current_node == "PLAN_APPROACH"


def test_advance_plan_approach_pass_through():
    e = _pick_engine()
    run = _started_run(e)
    e.advance(run, T0 + 1, _target(), _perception(), _act())  # -> PLAN_APPROACH
    old = e.advance(run, T0 + 2, _target(), _perception(), _act())
    assert old == PhaseId.PLAN_APPROACH
    assert run.current_node == "MOVE_PREGRASP"


def test_full_happy_path():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        grasp_distance_threshold_m=0.02,
        grasp_persistence_ms=0,
        close_gripper_duration_ms=0,
        verify_duration_ms=0,
        lift_duration_ms=0,
        skip_verify_grasp=True,
    )
    e = _pick_engine(cfg)
    run = _started_run(e)
    t = T0

    # SELECT_GRASP -> PLAN_APPROACH -> MOVE_PREGRASP
    t += 1
    e.advance(run, t, _target(), _perception(), _act())
    t += 1
    e.advance(run, t, _target(), _perception(), _act())
    assert run.current_node == "MOVE_PREGRASP"

    # MOVE_PREGRASP -> VISUAL_ALIGN
    t += 1
    e.advance(run, t, _target(0.10), _perception(), _act())
    assert run.current_node == "VISUAL_ALIGN"

    # VISUAL_ALIGN -> EXECUTE_APPROACH
    t += 1
    e.advance(run, t, _target(0.03), _perception(), _act())
    assert run.current_node == "EXECUTE_APPROACH"

    # EXECUTE_APPROACH -> CLOSE_GRIPPER
    t += 1
    e.advance(run, t, _target(0.01), _perception(), _act())
    assert run.current_node == "CLOSE_GRIPPER"

    # CLOSE_GRIPPER -> LIFT (skip_verify)
    t += 1
    e.advance(run, t, _target(), _perception(), _act())
    assert run.current_node == "LIFT"

    # LIFT -> DONE (success)
    t += 1
    e.advance(run, t, _target(), _perception(), _act())
    assert run.current_node == "DONE"
    assert run.outcome == SkillOutcomeState.SUCCESS
    assert not run.is_active


def test_abort():
    e = _pick_engine()
    run = _started_run(e)
    e.abort(run, T0 + 100)
    assert run.current_node == "DONE"
    assert run.outcome == SkillOutcomeState.FAILURE
    assert run.failure_code == SkillFailureCode.UNSAFE_ABORT


def test_fail():
    e = _pick_engine()
    run = _started_run(e)
    e.fail(run, T0 + 100, SkillFailureCode.PERCEPTION_LOST)
    assert run.current_node == "DONE"
    assert run.failure_code == SkillFailureCode.PERCEPTION_LOST


def test_sync_phase_forward():
    e = _pick_engine()
    run = _started_run(e)
    # Advance to MOVE_PREGRASP first
    e.advance(run, T0 + 1, _target(), _perception(), _act())
    e.advance(run, T0 + 2, _target(), _perception(), _act())
    assert run.current_node == "MOVE_PREGRASP"

    old = e.sync_phase(run, T0 + 3, PhaseId.EXECUTE_APPROACH)
    assert old == PhaseId.MOVE_PREGRASP
    assert run.current_node == "EXECUTE_APPROACH"


def test_sync_phase_ignores_backward():
    e = _pick_engine()
    run = _started_run(e)
    e.advance(run, T0 + 1, _target(), _perception(), _act())
    e.advance(run, T0 + 2, _target(), _perception(), _act())
    old = e.sync_phase(run, T0 + 3, PhaseId.IDLE)
    assert old is None
    assert run.current_node == "MOVE_PREGRASP"


def test_sync_phase_done_succeeds():
    e = _pick_engine()
    run = _started_run(e)
    e.advance(run, T0 + 1, _target(), _perception(), _act())
    e.advance(run, T0 + 2, _target(), _perception(), _act())
    old = e.sync_phase(run, T0 + 3, PhaseId.DONE)
    assert old == PhaseId.MOVE_PREGRASP
    assert run.outcome == SkillOutcomeState.SUCCESS


def test_transition_history_recorded():
    e = _pick_engine()
    run = _started_run(e)
    e.advance(run, T0 + 1, _target(), _perception(), _act())
    assert len(run.transition_history) == 1
    assert run.transition_history[0].from_node == "SELECT_GRASP"
    assert run.transition_history[0].to_node == "PLAN_APPROACH"


def test_needs_chunk():
    e = _pick_engine(_cfg(buffer_target_ms=200))
    run = _started_run(e)
    low = ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=100, buffer_low=True)
    high = ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=300, buffer_low=False)
    assert e.needs_chunk(run, low) is True
    assert e.needs_chunk(run, high) is False


def test_wrist_camera_active():
    cfg = _cfg(approach_align_threshold_m=0.15, execute_approach_threshold_m=0.05)
    e = _pick_engine(cfg)
    run = _started_run(e)
    assert not run.wrist_camera_active  # SELECT_GRASP

    e.advance(run, T0 + 1, _target(), _perception(), _act())  # PLAN_APPROACH
    e.advance(run, T0 + 2, _target(), _perception(), _act())  # MOVE_PREGRASP
    assert not run.wrist_camera_active

    e.advance(run, T0 + 3, _target(0.10), _perception(), _act())  # VISUAL_ALIGN
    assert run.wrist_camera_active


def test_reacquire_failed_guard():
    e = _pick_engine()
    run = _started_run(e)
    e.advance(run, T0 + 1, _target(), _perception(), _act())  # PLAN_APPROACH
    e.advance(run, T0 + 2, _target(), _perception(), _act())  # MOVE_PREGRASP

    failed_perc = PerceptionInfo(
        tracking_status=TrackingStatus.REACQUIRING,
        failure_code=PerceptionFailureCode.REACQUIRE_FAILED,
        reacquire_fail_count=3,
        vlm_job_pending=False,
    )
    old = e.advance(run, T0 + 3, _target(), failed_perc, _act())
    assert old == PhaseId.MOVE_PREGRASP
    assert run.current_node == "DONE"
    assert run.failure_code == SkillFailureCode.PERCEPTION_LOST


def test_already_held_guard_skips_to_done():
    """If the target object is already held, PICK should succeed immediately."""
    e = _pick_engine()
    run = _started_run(e)
    assert run.current_node == "SELECT_GRASP"

    # First advance with held_object_handle matching target
    old = e.advance(run, T0 + 1, _target(), _perception(), _act(), held_object_handle="obj-1")
    assert old == PhaseId.SELECT_GRASP
    assert run.current_node == "DONE"
    assert run.outcome == SkillOutcomeState.SUCCESS
    assert run.failure_code is None


def test_already_held_guard_ignores_different_object():
    """If a different object is held, PICK should proceed normally."""
    e = _pick_engine()
    run = _started_run(e)

    # Holding a different object — guard should not fire
    old = e.advance(run, T0 + 1, _target(), _perception(), _act(), held_object_handle="other-obj")
    # Normal SELECT_GRASP → PLAN_APPROACH transition (tracking is ok)
    assert old == PhaseId.SELECT_GRASP
    assert run.current_node == "PLAN_APPROACH"


def test_already_held_guard_ignores_none():
    """If nothing is held, PICK should proceed normally."""
    e = _pick_engine()
    run = _started_run(e)

    old = e.advance(run, T0 + 1, _target(), _perception(), _act(), held_object_handle=None)
    assert old == PhaseId.SELECT_GRASP
    assert run.current_node == "PLAN_APPROACH"
