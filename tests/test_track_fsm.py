"""Unit tests for TrackFSM — pure FSM for the TRACK skill."""

import pytest

from halo.contracts.enums import (
    PerceptionFailureCode,
    PhaseId,
    SkillFailureCode,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.snapshots import ActInfo, PerceptionInfo, TargetInfo
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.track_fsm import TrackFSM

_CFG = SkillRunnerConfig(acquiring_timeout_ms=5000)


def _tracking_perception() -> PerceptionInfo:
    return PerceptionInfo(
        tracking_status=TrackingStatus.TRACKING,
        failure_code=PerceptionFailureCode.OK,
        reacquire_fail_count=0,
        vlm_job_pending=False,
    )


def _idle_perception() -> PerceptionInfo:
    return PerceptionInfo(
        tracking_status=TrackingStatus.IDLE,
        failure_code=PerceptionFailureCode.OK,
        reacquire_fail_count=0,
        vlm_job_pending=False,
    )


def _target(handle: str = "cube-1") -> TargetInfo:
    return TargetInfo(
        handle=handle,
        hint_valid=True,
        confidence=0.9,
        obs_age_ms=5,
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, -0.1),
        distance_m=0.1,
    )


def _act() -> ActInfo:
    return ActInfo(status="IDLE", buffer_fill_ms=0, buffer_low=False, wrist_enabled=False)


def test_start_transitions_to_acquiring():
    fsm = TrackFSM(_CFG)
    fsm.start(1000, "cube-1")
    assert fsm.phase == PhaseId.ACQUIRING
    assert fsm.is_active


def test_advance_to_success():
    fsm = TrackFSM(_CFG)
    fsm.start(1000, "cube-1")
    old = fsm.advance(1500, _target("cube-1"), _tracking_perception(), _act())
    assert old == PhaseId.ACQUIRING
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.SUCCESS
    assert fsm.failure_code is None
    assert not fsm.is_active


def test_timeout_perception_lost():
    fsm = TrackFSM(_CFG)
    fsm.start(1000, "cube-1")
    # Advance with idle perception beyond timeout
    old = fsm.advance(1000 + _CFG.acquiring_timeout_ms, None, _idle_perception(), _act())
    assert old == PhaseId.ACQUIRING
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.FAILURE
    assert fsm.failure_code == SkillFailureCode.PERCEPTION_LOST


def test_timeout_target_mismatch():
    """Tracker is TRACKING with wrong handle at timeout → TARGET_MISMATCH."""
    fsm = TrackFSM(_CFG)
    fsm.start(1000, "cube-1")
    # Tracking established but for wrong handle, then timeout fires
    old = fsm.advance(
        1000 + _CFG.acquiring_timeout_ms,
        _target("cube-2"),
        _tracking_perception(),
        _act(),
    )
    assert old == PhaseId.ACQUIRING
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.FAILURE
    assert fsm.failure_code == SkillFailureCode.TARGET_MISMATCH


def test_handle_mismatch_blocks():
    fsm = TrackFSM(_CFG)
    fsm.start(1000, "cube-1")
    # Tracking established but for wrong handle
    old = fsm.advance(1500, _target("cube-2"), _tracking_perception(), _act())
    assert old is None
    assert fsm.phase == PhaseId.ACQUIRING
    assert fsm.is_active


def test_abort():
    fsm = TrackFSM(_CFG)
    fsm.start(1000, "cube-1")
    fsm.abort(1500)
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.FAILURE
    assert fsm.failure_code == SkillFailureCode.UNSAFE_ABORT


def test_fail_explicit():
    fsm = TrackFSM(_CFG)
    fsm.start(1000, "cube-1")
    fsm.fail(1500, SkillFailureCode.TIMEOUT)
    assert fsm.phase == PhaseId.DONE
    assert fsm.outcome == SkillOutcomeState.FAILURE
    assert fsm.failure_code == SkillFailureCode.TIMEOUT


def test_advance_noop_when_idle():
    fsm = TrackFSM(_CFG)
    old = fsm.advance(1000, None, _idle_perception(), _act())
    assert old is None


def test_start_raises_if_not_idle():
    fsm = TrackFSM(_CFG)
    fsm.start(1000, "cube-1")
    with pytest.raises(RuntimeError):
        fsm.start(2000, "cube-2")


def test_no_target_stays_in_acquiring():
    fsm = TrackFSM(_CFG)
    fsm.start(1000, "cube-1")
    old = fsm.advance(1500, None, _tracking_perception(), _act())
    assert old is None
    assert fsm.phase == PhaseId.ACQUIRING
