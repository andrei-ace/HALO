from __future__ import annotations

from dataclasses import dataclass, field

from halo.contracts.enums import CommandAckStatus, CommandType, SkillName


@dataclass(frozen=True)
class StartSkillPayload:
    skill_name: SkillName
    target_handle: str
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AbortSkillPayload:
    skill_run_id: str
    reason: str


@dataclass(frozen=True)
class OverrideTargetPayload:
    skill_run_id: str
    target_handle: str


@dataclass(frozen=True)
class DescribeScenePayload:
    reason: str


@dataclass(frozen=True)
class TrackObjectPayload:
    target_handle: str


CommandPayload = (
    StartSkillPayload | AbortSkillPayload | OverrideTargetPayload | DescribeScenePayload | TrackObjectPayload
)


@dataclass(frozen=True)
class CommandEnvelope:
    command_id: str  # UUID
    arm_id: str
    issued_at_ms: int
    type: CommandType
    payload: CommandPayload
    precondition_snapshot_id: str | None = None
    epoch: int | None = None  # lease epoch; None skips epoch check
    lease_token: str | None = None  # lease token; None skips token check


@dataclass(frozen=True)
class CommandAck:
    command_id: str
    status: CommandAckStatus
    reason: str | None = None
