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


# --- Initial state ---

def test_fsm_starts_in_reset():
    fsm = PickFSM(_cfg())
    assert fsm.phase == PhaseId.RESET
    assert not fsm.is_active


def test_start_transitions_to_approach():
    fsm = _started_fsm()
    assert fsm.phase == PhaseId.APPROACH_PREGRASP
    assert fsm.is_active
    assert fsm.outcome == SkillOutcomeState.IN_PROGRESS
    assert fsm.failure_code is None


def test_start_raises_if_not_in_reset():
    fsm = _started_fsm()
    with pytest.raises(RuntimeError):
        fsm.start(T0 + 100)


# --- APPROACH_PREGRASP ---

def test_approach_transitions_to_align_on_close_target():
    fsm = _started_fsm(_cfg(approach_align_threshold_m=0.15))
    old = fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())
    assert old == PhaseId.APPROACH_PREGRASP
    assert fsm.phase == PhaseId.ALIGN


def test_approach_stays_when_target_far():
    fsm = _started_fsm(_cfg(approach_align_threshold_m=0.15))
    old = fsm.advance(T0 + 1, _target(distance_m=0.5), _perception(), _act())
    assert old is None
    assert fsm.phase == PhaseId.APPROACH_PREGRASP


# --- ALIGN ---

def test_align_transitions_to_descend():
    fsm = _started_fsm(_cfg(approach_align_threshold_m=0.15, descend_threshold_m=0.05))
    # Move to ALIGN
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())
    assert fsm.phase == PhaseId.ALIGN

    # Now trigger DESCEND
    old = fsm.advance(T0 + 2, _target(distance_m=0.03), _perception(), _act())
    assert old == PhaseId.ALIGN
    assert fsm.phase == PhaseId.DESCEND_GRASP


# --- DESCEND_GRASP ---

def test_descend_grasp_close_triggers_deterministically():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        descend_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,  # immediate on first sub-threshold tick
    )
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())  # → ALIGN
    fsm.advance(T0 + 2, _target(distance_m=0.03), _perception(), _act())  # → DESCEND

    old = fsm.advance(T0 + 3, _target(distance_m=0.005), _perception(), _act())
    assert old == PhaseId.DESCEND_GRASP
    assert fsm.phase == PhaseId.CLOSE


def test_descend_grasp_resets_persistence_if_distance_increases():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        descend_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=500,
    )
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())  # → ALIGN
    fsm.advance(T0 + 2, _target(distance_m=0.03), _perception(), _act())  # → DESCEND

    # Below threshold — starts qualifying clock
    fsm.advance(T0 + 3, _target(distance_m=0.005), _perception(), _act())
    assert fsm._grasp_qualify_start_ms == T0 + 3

    # Distance bounces back above threshold — clock should reset
    fsm.advance(T0 + 4, _target(distance_m=0.05), _perception(), _act())
    assert fsm._grasp_qualify_start_ms is None
    assert fsm.phase == PhaseId.DESCEND_GRASP


# --- CLOSE ---

def test_close_transitions_to_verify_after_duration():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        descend_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_duration_ms=100,
        skip_verify_grasp=False,
    )
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())   # APPROACH→ALIGN
    fsm.advance(T0 + 2, _target(distance_m=0.03), _perception(), _act())   # ALIGN→DESCEND
    fsm.advance(T0 + 3, _target(distance_m=0.005), _perception(), _act())  # DESCEND→CLOSE
    assert fsm.phase == PhaseId.CLOSE
    phase_start = fsm.phase_start_ms

    old = fsm.advance(phase_start + 100, _target(distance_m=0.005), _perception(), _act())
    assert old == PhaseId.CLOSE
    assert fsm.phase == PhaseId.VERIFY_GRASP


def test_close_transitions_to_lift_when_skip_verify():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        descend_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_duration_ms=100,
        skip_verify_grasp=True,
    )
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(T0 + 2, _target(distance_m=0.03), _perception(), _act())
    fsm.advance(T0 + 3, _target(distance_m=0.005), _perception(), _act())
    phase_start = fsm.phase_start_ms

    old = fsm.advance(phase_start + 100, _target(), _perception(), _act())
    assert old == PhaseId.CLOSE
    assert fsm.phase == PhaseId.LIFT


def test_skip_verify_grasp_when_configured():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        descend_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_duration_ms=0,
        skip_verify_grasp=True,
    )
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(T0 + 2, _target(distance_m=0.03), _perception(), _act())
    fsm.advance(T0 + 3, _target(distance_m=0.005), _perception(), _act())
    # CLOSE with duration=0 should immediately go to LIFT (skip verify)
    phase_start = fsm.phase_start_ms
    fsm.advance(phase_start + 0, _target(), _perception(), _act())
    assert fsm.phase == PhaseId.LIFT


# --- VERIFY_GRASP ---

def test_verify_grasp_transitions_to_lift():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        descend_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_duration_ms=0,
        verify_duration_ms=200,
        skip_verify_grasp=False,
    )
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(T0 + 2, _target(distance_m=0.03), _perception(), _act())
    fsm.advance(T0 + 3, _target(distance_m=0.005), _perception(), _act())
    # CLOSE→VERIFY (duration=0)
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
        descend_threshold_m=0.05,
        grasp_distance_threshold_m=0.01,
        grasp_persistence_ms=0,
        close_duration_ms=0,
        verify_duration_ms=0,
        lift_duration_ms=500,
        skip_verify_grasp=False,
    )
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(T0 + 2, _target(distance_m=0.03), _perception(), _act())
    fsm.advance(T0 + 3, _target(distance_m=0.005), _perception(), _act())
    # CLOSE→VERIFY
    phase_start = fsm.phase_start_ms
    fsm.advance(phase_start, _target(), _perception(), _act())
    # VERIFY→LIFT
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

def test_approach_timeout_causes_failure():
    cfg = _cfg(approach_timeout_ms=1000)
    fsm = _started_fsm(cfg)
    old = fsm.advance(T0 + 1000, _target(distance_m=0.5), _perception(), _act())
    assert old == PhaseId.APPROACH_PREGRASP
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.FAILURE
    assert fsm.failure_code == SkillFailureCode.NO_PROGRESS


def test_align_timeout_causes_failure():
    cfg = _cfg(approach_align_threshold_m=0.15, align_timeout_ms=500)
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())  # → ALIGN
    assert fsm.phase == PhaseId.ALIGN
    align_start = fsm.phase_start_ms

    old = fsm.advance(align_start + 500, _target(distance_m=0.10), _perception(), _act())
    assert old == PhaseId.ALIGN
    assert fsm.phase == PhaseId.DONE
    assert fsm.failure_code == SkillFailureCode.NO_PROGRESS


def test_descend_timeout_causes_no_grasp_failure():
    cfg = _cfg(
        approach_align_threshold_m=0.15,
        descend_threshold_m=0.05,
        descend_timeout_ms=800,
    )
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())
    fsm.advance(T0 + 2, _target(distance_m=0.03), _perception(), _act())
    assert fsm.phase == PhaseId.DESCEND_GRASP
    descend_start = fsm.phase_start_ms

    old = fsm.advance(descend_start + 800, _target(distance_m=0.03), _perception(), _act())
    assert old == PhaseId.DESCEND_GRASP
    assert fsm.phase == PhaseId.DONE
    assert fsm.failure_code == SkillFailureCode.NO_GRASP


# --- Target loss & recovery ---

def test_lost_target_from_approach_triggers_recovery():
    cfg = _cfg(no_target_tolerance_ms=500)
    fsm = _started_fsm(cfg)

    # First call: no_target_start_ms gets set to T0+1, elapsed=0 → stay
    fsm.advance(T0 + 1, _no_target(), _perception(), _act())
    assert fsm.phase == PhaseId.APPROACH_PREGRASP

    # Second call: elapsed = (T0+1+500) - (T0+1) = 500 >= 500 → recovery
    old = fsm.advance(T0 + 1 + 500, _no_target(), _perception(), _act())
    assert old == PhaseId.APPROACH_PREGRASP
    assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH


def test_lost_target_from_align_triggers_recovery():
    cfg = _cfg(approach_align_threshold_m=0.15, no_target_tolerance_ms=500)
    fsm = _started_fsm(cfg)
    fsm.advance(T0 + 1, _target(distance_m=0.10), _perception(), _act())  # → ALIGN

    fsm.advance(T0 + 2, _no_target(), _perception(), _act())
    assert fsm.phase == PhaseId.ALIGN

    old = fsm.advance(T0 + 2 + 500, _no_target(), _perception(), _act())
    assert old == PhaseId.ALIGN
    assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH


def test_recovery_returns_to_approach_after_wait():
    cfg = _cfg(no_target_tolerance_ms=500, recover_wait_ms=200, max_reacquire_attempts=3)
    fsm = _started_fsm(cfg)

    # Trigger recovery
    fsm.advance(T0 + 1, _no_target(), _perception(), _act())
    fsm.advance(T0 + 1 + 500, _no_target(), _perception(), _act())
    assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH
    recover_start = fsm.phase_start_ms

    # Wait is over → return to APPROACH
    old = fsm.advance(recover_start + 200, _target(distance_m=0.5), _perception(), _act())
    assert old == PhaseId.RECOVER_RETRY_APPROACH
    assert fsm.phase == PhaseId.APPROACH_PREGRASP
    assert fsm._reacquire_count == 1


def test_max_reacquire_causes_failure():
    cfg = _cfg(no_target_tolerance_ms=500, recover_wait_ms=200, max_reacquire_attempts=2)
    fsm = _started_fsm(cfg)

    def _do_recovery_cycle(base_ms: int) -> int:
        """Drive one full recovery cycle. Returns time after re-entering APPROACH."""
        # Trigger target loss then recovery
        fsm.advance(base_ms + 1, _no_target(), _perception(), _act())
        fsm.advance(base_ms + 1 + 500, _no_target(), _perception(), _act())
        assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH
        recover_start = fsm.phase_start_ms
        fsm.advance(recover_start + 200, _target(distance_m=0.5), _perception(), _act())
        return recover_start + 200

    t = T0
    t = _do_recovery_cycle(t)   # reacquire_count → 1, back to APPROACH
    assert fsm._reacquire_count == 1
    t = _do_recovery_cycle(t)   # reacquire_count → 2, back to APPROACH
    assert fsm._reacquire_count == 2

    # Third recovery: count would become 3 > 2 → FAIL
    fsm.advance(t + 1, _no_target(), _perception(), _act())
    fsm.advance(t + 1 + 500, _no_target(), _perception(), _act())
    assert fsm.phase == PhaseId.RECOVER_RETRY_APPROACH
    recover_start = fsm.phase_start_ms
    old = fsm.advance(recover_start + 200, _target(distance_m=0.5), _perception(), _act())
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
