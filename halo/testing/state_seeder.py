"""Reusable state-seeding helpers for HALO tests.

Extracts common patterns from test files into shared factory functions.
"""

from __future__ import annotations

from halo.contracts.enums import (
    ActStatus,
    PerceptionFailureCode,
    PhaseId,
    SkillName,
    TrackingStatus,
)
from halo.contracts.snapshots import (
    ActInfo,
    PerceptionInfo,
    SkillInfo,
    TargetInfo,
)
from halo.runtime.runtime import HALORuntime


def make_target(
    handle: str = "obj-1",
    distance_m: float = 0.5,
    confidence: float = 0.9,
    hint_valid: bool = True,
    obs_age_ms: int = 10,
    time_skew_ms: int = 0,
    delta_xyz_ee: tuple[float, float, float] | None = None,
    center_px: tuple[float, float] | None = None,
    bbox_xywh: tuple[int, int, int, int] | None = None,
) -> TargetInfo:
    """Build a TargetInfo with sensible defaults."""
    if delta_xyz_ee is None:
        delta_xyz_ee = (0.0, 0.0, distance_m)
    return TargetInfo(
        handle=handle,
        hint_valid=hint_valid,
        confidence=confidence,
        obs_age_ms=obs_age_ms,
        time_skew_ms=time_skew_ms,
        delta_xyz_ee=delta_xyz_ee,
        distance_m=distance_m,
        center_px=center_px,
        bbox_xywh=bbox_xywh,
    )


def make_perception(
    tracking_status: TrackingStatus = TrackingStatus.TRACKING,
    failure_code: PerceptionFailureCode = PerceptionFailureCode.OK,
    reacquire_fail_count: int = 0,
    vlm_job_pending: bool = False,
) -> PerceptionInfo:
    """Build a PerceptionInfo with sensible defaults."""
    return PerceptionInfo(
        tracking_status=tracking_status,
        failure_code=failure_code,
        reacquire_fail_count=reacquire_fail_count,
        vlm_job_pending=vlm_job_pending,
    )


def make_act(
    status: ActStatus = ActStatus.IDLE,
    buffer_fill_ms: int = 0,
    buffer_low: bool | None = None,
    wrist_enabled: bool = False,
) -> ActInfo:
    """Build an ActInfo with sensible defaults.

    If *buffer_low* is not specified, it defaults to True when fill_ms == 0.
    """
    if buffer_low is None:
        buffer_low = buffer_fill_ms == 0
    return ActInfo(
        status=status,
        buffer_fill_ms=buffer_fill_ms,
        buffer_low=buffer_low,
        wrist_enabled=wrist_enabled,
    )


def make_skill(
    name: SkillName = SkillName.PICK,
    skill_run_id: str = "run-test-1",
    phase: PhaseId = PhaseId.IDLE,
) -> SkillInfo:
    """Build a SkillInfo with sensible defaults."""
    return SkillInfo(name=name, skill_run_id=skill_run_id, phase=phase)


async def seed_store(
    rt: HALORuntime,
    arm_id: str = "arm0",
    *,
    target: TargetInfo | None = None,
    perception: PerceptionInfo | None = None,
    act: ActInfo | None = None,
    skill: SkillInfo | None = None,
    build_snapshot: bool = True,
) -> None:
    """Seed the RuntimeStateStore with the given state.

    Only updates fields for which a value was explicitly passed.
    If *build_snapshot* is True (default), also triggers a snapshot build
    so the state is visible to ``get_latest_runtime_snapshot()``.
    """
    store = rt.store
    if target is not None:
        await store.update_target(arm_id, target)
    if perception is not None:
        await store.update_perception(arm_id, perception)
    if act is not None:
        await store.update_act(arm_id, act)
    if skill is not None:
        await store.update_skill(arm_id, skill)
    if build_snapshot:
        recent = rt.bus.get_recent_events(arm_id)
        await store.build_and_cache_snapshot(arm_id, recent)
