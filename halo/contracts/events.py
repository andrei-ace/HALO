from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EventType(StrEnum):
    COMMAND_ACCEPTED = "COMMAND_ACCEPTED"
    COMMAND_REJECTED = "COMMAND_REJECTED"
    SKILL_STARTED = "SKILL_STARTED"
    SKILL_SUCCEEDED = "SKILL_SUCCEEDED"
    SKILL_FAILED = "SKILL_FAILED"
    PHASE_ENTER = "PHASE_ENTER"
    PHASE_EXIT = "PHASE_EXIT"
    PERCEPTION_FAILURE = "PERCEPTION_FAILURE"
    PERCEPTION_RECOVERED = "PERCEPTION_RECOVERED"
    SCENE_DESCRIBED = "SCENE_DESCRIBED"
    TARGET_ACQUIRED = "TARGET_ACQUIRED"
    SAFETY_REFLEX_TRIGGERED = "SAFETY_REFLEX_TRIGGERED"
    SAFETY_RECOVERED = "SAFETY_RECOVERED"


@dataclass(frozen=True)
class EventEnvelope:
    event_id: str
    type: EventType
    ts_ms: int
    arm_id: str
    data: dict = field(default_factory=dict)
