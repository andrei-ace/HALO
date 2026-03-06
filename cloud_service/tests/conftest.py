"""Shared fixtures for cloud_service tests."""

from __future__ import annotations

from halo.contracts.enums import (
    ActStatus,
    PerceptionFailureCode,
    SafetyState,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.snapshots import (
    ActInfo,
    OutcomeInfo,
    PerceptionInfo,
    PlannerSnapshot,
    ProgressInfo,
    SafetyInfo,
    TargetInfo,
)


def idle_snapshot() -> PlannerSnapshot:
    return PlannerSnapshot(
        snapshot_id="snap-001",
        ts_ms=1000,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle=None,
            hint_valid=False,
            confidence=0.0,
            obs_age_ms=0,
            time_skew_ms=0,
            delta_xyz_ee=(0.0, 0.0, 0.0),
            distance_m=0.0,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.LOST,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False),
        safety=SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=()),
        command_acks=(),
        recent_events=(),
        held_object_handle=None,
    )
