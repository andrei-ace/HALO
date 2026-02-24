from __future__ import annotations

from dataclasses import dataclass

from halo.contracts.commands import CommandAck
from halo.contracts.enums import (
    ActStatus,
    PerceptionFailureCode,
    PhaseId,
    SafetyReflexReason,
    SafetyState,
    SkillFailureCode,
    SkillName,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.events import EventEnvelope


@dataclass(frozen=True)
class SkillInfo:
    name: SkillName
    skill_run_id: str
    phase: PhaseId


@dataclass(frozen=True)
class TargetInfo:
    handle: str
    hint_valid: bool
    confidence: float
    obs_age_ms: int
    time_skew_ms: int
    delta_xyz_ee: tuple[float, float, float]
    distance_m: float


@dataclass(frozen=True)
class PerceptionInfo:
    tracking_status: TrackingStatus
    failure_code: PerceptionFailureCode
    reacquire_fail_count: int
    vlm_job_pending: bool


@dataclass(frozen=True)
class ActInfo:
    status: ActStatus
    buffer_fill_ms: int
    buffer_low: bool


@dataclass(frozen=True)
class ProgressInfo:
    elapsed_ms: int
    no_progress_ms: int
    delta_distance: float


@dataclass(frozen=True)
class OutcomeInfo:
    state: SkillOutcomeState
    reason_code: SkillFailureCode | None
    needs_verify: bool


@dataclass(frozen=True)
class SafetyInfo:
    state: SafetyState
    reflex_active: bool
    reason_codes: tuple[SafetyReflexReason, ...]


@dataclass(frozen=True)
class PlannerSnapshot:
    snapshot_id: str
    ts_ms: int
    arm_id: str
    skill: SkillInfo | None
    target: TargetInfo | None
    perception: PerceptionInfo
    act: ActInfo
    progress: ProgressInfo
    outcome: OutcomeInfo
    safety: SafetyInfo
    command_acks: tuple[CommandAck, ...]
    recent_events: tuple[EventEnvelope, ...]
