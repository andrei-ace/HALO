"""Tests for PickFSM: pure synchronous state machine."""

import pytest

from halo.contracts.enums import (
    ActStatus,
    PerceptionFailureCode,
    PhaseId,
    SkillFailureCode,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.snapshots import ActInfo, PerceptionInfo, TargetInfo
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.fsm import PickFSM

T0 = 1000  # arbitrary start ms


def _cfg(**kwargs) -> SkillRunnerConfig:
    kwargs.setdefault("runner_rate_hz", 10.0)
    return SkillRunnerConfig(**kwargs)


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


def _no_target() -> TargetInfo:
    return _target(distance_m=0.5, hint_valid=False)


def _perception() -> PerceptionInfo:
    return PerceptionInfo(
        tracking_status=TrackingStatus.TRACKING,
        failure_code=PerceptionFailureCode.OK,
        reacquire_fail_count=0,
        vlm_job_pending=False,
    )


def _act(fill_ms: int = 300) -> ActInfo:
    return ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=fill_ms, buffer_low=False)


def _started_fsm(cfg: SkillRunnerConfig | None = None) -> PickFSM:
    if cfg is None:
        cfg = _cfg()
    fsm = PickFSM(cfg)
    fsm.start(T0)
    return fsm


def _advance_to_move_pregrasp(fsm: PickFSM, t: int) -> int:
    """Advance past SELECT_GRASP and PLAN_APPROACH (v0: immediate pass-through).
    Returns the time after transitions."""
    # SELECT_GRASP -> PLAN_APPROACH (immediate)
    fsm.advance(t, _target(), _perception(), _act())
    assert fsm.phase == PhaseId.PLAN_APPROACH
    # PLAN_APPROACH -> MOVE_PREGRASP (immediate)
    fsm.advance(t + 1, _target(), _perception(), _act())
    assert fsm.phase == PhaseId.MOVE_PREGRASP
    return t + 1


# --- Initial state ---


def test_fsm_starts_in_idle():
    fsm = PickFSM(_cfg())
    assert fsm.phase == PhaseId.IDLE
    assert not fsm.is_active


def test_start_transitions_to_select_grasp():
    fsm = _started_fsm()
    assert fsm.phase == PhaseId.SELECT_GRASP
    assert fsm.is_active
    assert fsm.outcome == SkillOutcomeState.IN_PROGRESS
    assert fsm.failure_code is None


def test_start_raises_if_not_in_idle():
    fsm = _started_fsm()
    with pytest.raises(RuntimeError):
        fsm.start(T0 + 100)


# --- SELECT_GRASP -> PLAN_APPROACH -> MOVE_PREGRASP (v0: immediate) ---


def test_select_grasp_immediately_transitions():
    fsm = _started_fsm()
    old = fsm.advance(T0 + 1, _target(), _perception(), _act())
    assert old == PhaseId.SELECT_GRASP
    assert fsm.phase == PhaseId.PLAN_APPROACH


def test_plan_approach_immediately_transitions():
    fsm = _started_fsm()
    fsm.advance(T0 + 1, _target(), _perception(), _act())  # -> PLAN_APPROACH
    old = fsm.advance(T0 + 2, _target(), _perception(), _act())
    assert old == PhaseId.PLAN_APPROACH
    assert fsm.phase == PhaseId.MOVE_PREGRASP


# --- MOVE_PREGRASP ---


def test_move_pregrasp_transitions_to_visual_align_on_close_target():
    cfg = _cfg(approach_align_threshold_m=0.15)
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    old = fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())
    assert old == PhaseId.MOVE_PREGRASP
    assert fsm.phase == PhaseId.VISUAL_ALIGN


def test_move_pregrasp_stays_when_target_far():
    cfg = _cfg(approach_align_threshold_m=0.15)
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    old = fsm.advance(t + 1, _target(distance_m=0.5), _perception(), _act())
    assert old is None
    assert fsm.phase == PhaseId.MOVE_PREGRASP


# --- VISUAL_ALIGN ---


def test_visual_align_transitions_to_execute_approach():
    cfg = _cfg(approach_align_threshold_m=0.15, execute_approach_threshold_m=0.05)
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    # -> VISUAL_ALIGN
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())
    assert fsm.phase == PhaseId.VISUAL_ALIGN

    # -> EXECUTE_APPROACH
    old = fsm.advance(t + 2, _target(distance_m=0.03), _perception(), _act())
    assert old == PhaseId.VISUAL_ALIGN
    assert fsm.phase == PhaseId.EXECUTE_APPROACH


# --- EXECUTE_APPROACH ---


def test_execute_approach_close_triggers_deterministically():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
    )
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())  # -> VISUAL_ALIGN
    fsm.advance(t + 2, _target(distance_m=0.03), _perception(), _act())  # -> EXECUTE_APPROACH

    old = fsm.advance(t + 3, _target(distance_m=0.005), _perception(), _act())
    assert old == PhaseId.EXECUTE_APPROACH
    assert fsm.phase == PhaseId.CLOSE_GRIPPER


def test_execute_approach_resets_persistence_if_distance_increases():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=500,
    )
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())  # -> VISUAL_ALIGN
    fsm.advance(t + 2, _target(distance_m=0.03), _perception(), _act())  # -> EXECUTE_APPROACH

    # Below threshold — starts qualifying clock
    fsm.advance(t + 3, _target(distance_m=0.005), _perception(), _act())
    assert fsm._grasp_qualify_start_ms == t + 3

    # Distance bounces back above threshold — clock should reset
    fsm.advance(t + 4, _target(distance_m=0.05), _perception(), _act())
    assert fsm._grasp_qualify_start_ms is None
    assert fsm.phase == PhaseId.EXECUTE_APPROACH


# --- CLOSE_GRIPPER ---


def test_close_gripper_transitions_to_verify_after_duration():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_gripper_duration_ms=100,
        skip_verify_grasp=False,
    )
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())  # -> VISUAL_ALIGN
    fsm.advance(t + 2, _target(distance_m=0.03), _perception(), _act())  # -> EXECUTE_APPROACH
    fsm.advance(t + 3, _target(distance_m=0.005), _perception(), _act())  # -> CLOSE_GRIPPER
    assert fsm.phase == PhaseId.CLOSE_GRIPPER
    phase_start = fsm.phase_start_ms

    old = fsm.advance(phase_start + 100, _target(distance_m=0.005), _perception(), _act())
    assert old == PhaseId.CLOSE_GRIPPER
    assert fsm.phase == PhaseId.VERIFY_GRASP


def test_close_gripper_transitions_to_lift_when_skip_verify():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_gripper_duration_ms=100,
        skip_verify_grasp=True,
    )
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(t + 2, _target(distance_m=0.03), _perception(), _act())
    fsm.advance(t + 3, _target(distance_m=0.005), _perception(), _act())
    phase_start = fsm.phase_start_ms

    old = fsm.advance(phase_start + 100, _target(), _perception(), _act())
    assert old == PhaseId.CLOSE_GRIPPER
    assert fsm.phase == PhaseId.LIFT


def test_skip_verify_grasp_when_configured():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_gripper_duration_ms=0,
        skip_verify_grasp=True,
    )
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(t + 2, _target(distance_m=0.03), _perception(), _act())
    fsm.advance(t + 3, _target(distance_m=0.005), _perception(), _act())
    # CLOSE_GRIPPER with duration=0 should immediately go to LIFT (skip verify)
    phase_start = fsm.phase_start_ms
    fsm.advance(phase_start + 0, _target(), _perception(), _act())
    assert fsm.phase == PhaseId.LIFT


# --- VERIFY_GRASP ---


def test_verify_grasp_transitions_to_lift():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_gripper_duration_ms=0,
        verify_duration_ms=200,
        skip_verify_grasp=False,
    )
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(t + 2, _target(distance_m=0.03), _perception(), _act())
    fsm.advance(t + 3, _target(distance_m=0.005), _perception(), _act())
    # CLOSE_GRIPPER->VERIFY (duration=0)
    phase_start = fsm.phase_start_ms
    fsm.advance(phase_start, _target(), _perception(), _act())
    assert fsm.phase == PhaseId.VERIFY_GRASP

    verify_start = fsm.phase_start_ms
    old = fsm.advance(verify_start + 200, _target(), _perception(), _act())
    assert old == PhaseId.VERIFY_GRASP
    assert fsm.phase == PhaseId.LIFT


# --- LIFT ---


def test_lift_transitions_to_done_with_success():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_gripper_duration_ms=0,
        verify_duration_ms=0,
        lift_duration_ms=500,
        skip_verify_grasp=False,
    )
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(t + 2, _target(distance_m=0.03), _perception(), _act())
    fsm.advance(t + 3, _target(distance_m=0.005), _perception(), _act())
    # CLOSE_GRIPPER->VERIFY
    phase_start = fsm.phase_start_ms
    fsm.advance(phase_start, _target(), _perception(), _act())
    # VERIFY->LIFT
    verify_start = fsm.phase_start_ms
    fsm.advance(verify_start, _target(), _perception(), _act())
    assert fsm.phase == PhaseId.LIFT

    lift_start = fsm.phase_start_ms
    old = fsm.advance(lift_start + 500, _target(), _perception(), _act())
    assert old == PhaseId.LIFT
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.SUCCESS
    assert not fsm.is_active


# --- Timeouts ---


def test_move_pregrasp_timeout_causes_failure():
    cfg = _cfg(move_pregrasp_timeout_ms=1000)
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    old = fsm.advance(t + 1000, _target(distance_m=0.5), _perception(), _act())
    assert old == PhaseId.MOVE_PREGRASP
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.FAILURE
    assert fsm.failure_code == SkillFailureCode.NO_PROGRESS


def test_visual_align_timeout_causes_failure():
    cfg = _cfg(approach_align_threshold_m=0.15, visual_align_timeout_ms=500)
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())  # -> VISUAL_ALIGN
    assert fsm.phase == PhaseId.VISUAL_ALIGN
    align_start = fsm.phase_start_ms

    old = fsm.advance(align_start + 500, _target(distance_m=0.10), _perception(), _act())
    assert old == PhaseId.VISUAL_ALIGN
    assert fsm.phase == PhaseId.DONE
    assert fsm.failure_code == SkillFailureCode.NO_PROGRESS


def test_execute_approach_timeout_causes_no_grasp_failure():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        execute_approach_threshold_m=0.05,
        execute_approach_timeout_ms=800,
    )
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(t + 2, _target(distance_m=0.03), _perception(), _act())
    assert fsm.phase == PhaseId.EXECUTE_APPROACH
    descend_start = fsm.phase_start_ms

    old = fsm.advance(descend_start + 800, _target(distance_m=0.03), _perception(), _act())
    assert old == PhaseId.EXECUTE_APPROACH
    assert fsm.phase == PhaseId.DONE
    assert fsm.failure_code == SkillFailureCode.NO_GRASP


# --- Target loss & recovery ---


def test_lost_target_from_move_pregrasp_triggers_recovery():
    cfg = _cfg(no_target_tolerance_ms=500)
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)

    # First call: no_target_start_ms gets set, elapsed=0 -> stay
    fsm.advance(t + 1, _no_target(), _perception(), _act())
    assert fsm.phase == PhaseId.MOVE_PREGRASP

    # Second call: elapsed >= 500 -> recovery
    old = fsm.advance(t + 1 + 500, _no_target(), _perception(), _act())
    assert old == PhaseId.MOVE_PREGRASP
    assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH


def test_lost_target_from_visual_align_triggers_recovery():
    cfg = _cfg(approach_align_threshold_m=0.15, no_target_tolerance_ms=500)
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(t + 1, _target(distance_m=0.10), _perception(), _act())  # -> VISUAL_ALIGN

    fsm.advance(t + 2, _no_target(), _perception(), _act())
    assert fsm.phase == PhaseId.VISUAL_ALIGN

    old = fsm.advance(t + 2 + 500, _no_target(), _perception(), _act())
    assert old == PhaseId.VISUAL_ALIGN
    assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH


def test_recovery_returns_to_move_pregrasp_after_wait():
    cfg = _cfg(no_target_tolerance_ms=500, recover_wait_ms=200, max_reacquire_attempts=3)
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)

    # Trigger recovery
    fsm.advance(t + 1, _no_target(), _perception(), _act())
    fsm.advance(t + 1 + 500, _no_target(), _perception(), _act())
    assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH
    recover_start = fsm.phase_start_ms

    # Wait is over -> return to MOVE_PREGRASP
    old = fsm.advance(recover_start + 200, _target(distance_m=0.5), _perception(), _act())
    assert old == PhaseId.RECOVER_RETRY_APPROACH
    assert fsm.phase == PhaseId.MOVE_PREGRASP
    assert fsm._reacquire_count == 1


def test_max_reacquire_causes_failure():
    cfg = _cfg(no_target_tolerance_ms=500, recover_wait_ms=200, max_reacquire_attempts=2)
    fsm = _started_fsm(cfg)
    t = _advance_to_move_pregrasp(fsm, T0 + 1)

    def _do_recovery_cycle(base_ms: int) -> int:
        """Drive one full recovery cycle. Returns time after re-entering MOVE_PREGRASP."""
        fsm.advance(base_ms + 1, _no_target(), _perception(), _act())
        fsm.advance(base_ms + 1 + 500, _no_target(), _perception(), _act())
        assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH
        recover_start = fsm.phase_start_ms
        fsm.advance(recover_start + 200, _target(distance_m=0.5), _perception(), _act())
        return recover_start + 200

    t = _do_recovery_cycle(t)  # reacquire_count -> 1
    assert fsm._reacquire_count == 1
    t = _do_recovery_cycle(t)  # reacquire_count -> 2
    assert fsm._reacquire_count == 2

    # Third recovery: count would become 3 > 2 -> FAIL
    fsm.advance(t + 1, _no_target(), _perception(), _act())
    fsm.advance(t + 1 + 500, _no_target(), _perception(), _act())
    assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH
    recover_start = fsm.phase_start_ms
    fsm.advance(recover_start + 200, _target(distance_m=0.5), _perception(), _act())
    assert fsm.phase == PhaseId.DONE
    assert fsm.failure_code == SkillFailureCode.TIMEOUT


# --- Abort ---


def test_abort_sets_failure_unsafe_abort():
    fsm = _started_fsm()
    fsm.abort(T0 + 100)
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.FAILURE
    assert fsm.failure_code == SkillFailureCode.UNSAFE_ABORT
    assert not fsm.is_active


def test_abort_idempotent_on_done():
    fsm = _started_fsm()
    fsm.abort(T0 + 100)
    # Second abort should not raise or change state
    fsm.abort(T0 + 200)
    assert fsm.phase == PhaseId.DONE
    assert fsm._phase_start_ms == T0 + 100  # unchanged


# --- needs_chunk ---


def test_needs_chunk_when_buffer_low():
    cfg = _cfg(buffer_target_ms=200)
    fsm = _started_fsm(cfg)
    act = ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=100, buffer_low=True)
    assert fsm.needs_chunk(act) is True


def test_no_chunk_needed_when_buffer_full():
    cfg = _cfg(buffer_target_ms=200)
    fsm = _started_fsm(cfg)
    act = ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=300, buffer_low=False)
    assert fsm.needs_chunk(act) is False


# --- Wrist camera ---


# --- sync_phase (sim mode) ---


def test_sync_phase_transitions_on_new_phase():
    fsm = _started_fsm()
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    old = fsm.sync_phase(t + 1, PhaseId.EXECUTE_APPROACH)
    assert old == PhaseId.MOVE_PREGRASP
    assert fsm.phase == PhaseId.EXECUTE_APPROACH


def test_sync_phase_returns_none_on_same_phase():
    fsm = _started_fsm()
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    old = fsm.sync_phase(t + 1, PhaseId.MOVE_PREGRASP)
    assert old is None
    assert fsm.phase == PhaseId.MOVE_PREGRASP


def test_sync_phase_done_sets_success():
    fsm = _started_fsm()
    _advance_to_move_pregrasp(fsm, T0 + 1)
    old = fsm.sync_phase(T0 + 100, PhaseId.DONE)
    assert old == PhaseId.MOVE_PREGRASP
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.SUCCESS


def test_sync_phase_noop_when_inactive():
    fsm = PickFSM(_cfg())  # IDLE, not started
    old = fsm.sync_phase(T0, PhaseId.EXECUTE_APPROACH)
    assert old is None
    assert fsm.phase == PhaseId.IDLE


def test_sync_phase_full_sequence():
    fsm = _started_fsm()
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    phases = [
        PhaseId.EXECUTE_APPROACH,
        PhaseId.CLOSE_GRIPPER,
        PhaseId.LIFT,
        PhaseId.DONE,
    ]
    for phase in phases:
        t += 1
        old = fsm.sync_phase(t, phase)
        assert old is not None
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.SUCCESS


def test_sync_phase_repeated_phase_returns_none():
    fsm = _started_fsm()
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.sync_phase(t + 1, PhaseId.EXECUTE_APPROACH)
    # Same phase again
    old = fsm.sync_phase(t + 2, PhaseId.EXECUTE_APPROACH)
    assert old is None
    assert fsm.phase == PhaseId.EXECUTE_APPROACH


def test_sync_phase_ignores_backward_transition():
    """Backward phase (e.g. IDLE while in SELECT_GRASP) must be ignored."""
    fsm = _started_fsm()
    assert fsm.phase == PhaseId.SELECT_GRASP
    old = fsm.sync_phase(T0 + 10, PhaseId.IDLE)
    assert old is None
    assert fsm.phase == PhaseId.SELECT_GRASP
    assert fsm.is_active


def test_sync_phase_ignores_backward_from_later_phase():
    """When FSM is in EXECUTE_APPROACH, MOVE_PREGRASP must be ignored."""
    fsm = _started_fsm()
    t = _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.sync_phase(t + 1, PhaseId.EXECUTE_APPROACH)
    assert fsm.phase == PhaseId.EXECUTE_APPROACH
    old = fsm.sync_phase(t + 2, PhaseId.MOVE_PREGRASP)
    assert old is None
    assert fsm.phase == PhaseId.EXECUTE_APPROACH


# --- Wrist camera ---


def test_advance_fails_on_reacquire_failed():
    """REACQUIRE_FAILED perception failure immediately aborts the skill."""
    fsm = _started_fsm()
    t = _advance_to_move_pregrasp(fsm, T0 + 1)

    failed_perception = PerceptionInfo(
        tracking_status=TrackingStatus.REACQUIRING,
        failure_code=PerceptionFailureCode.REACQUIRE_FAILED,
        reacquire_fail_count=3,
        vlm_job_pending=False,
    )
    old = fsm.advance(t + 1, _target(), failed_perception, _act())
    assert old == PhaseId.MOVE_PREGRASP
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.FAILURE
    assert fsm.failure_code == SkillFailureCode.PERCEPTION_LOST


def test_advance_ok_on_non_fatal_perception_failure():
    """Non-REACQUIRE_FAILED perception codes do not abort."""
    fsm = _started_fsm()
    t = _advance_to_move_pregrasp(fsm, T0 + 1)

    occluded = PerceptionInfo(
        tracking_status=TrackingStatus.TRACKING,
        failure_code=PerceptionFailureCode.OCCLUDED,
        reacquire_fail_count=0,
        vlm_job_pending=False,
    )
    fsm.advance(t + 1, _target(), occluded, _act())
    # Should NOT abort — still in a normal phase
    assert fsm.phase != PhaseId.DONE


def test_wrist_camera_active_in_correct_phases():
    fsm = PickFSM(_cfg())
    assert not fsm.wrist_camera_active  # IDLE

    fsm.start(T0)
    assert not fsm.wrist_camera_active  # SELECT_GRASP

    # Advance to VISUAL_ALIGN
    cfg = _cfg(approach_align_threshold_m=0.15)
    fsm = _started_fsm(cfg)
    _advance_to_move_pregrasp(fsm, T0 + 1)
    fsm.advance(T0 + 10, _target(distance_m=0.10), _perception(), _act())
    assert fsm.phase == PhaseId.VISUAL_ALIGN
    assert fsm.wrist_camera_active
