from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from halo.contracts.enums import WRIST_ACTIVE_PHASES, PhaseId, SkillFailureCode, SkillName, SkillOutcomeState
from halo.services.skill_runner_service.graph import FsmGraph


class NodeStatus(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TransitionRecord:
    from_node: str
    to_node: str
    ts_ms: int
    trigger: str


@dataclass(frozen=True)
class QueuedSkill:
    skill_name: SkillName
    skill_run_id: str
    target_handle: str
    variant: str
    options: dict
    enqueued_at_ms: int


@dataclass
class SkillRun:
    skill_run_id: str
    skill_name: SkillName
    variant: str
    target_handle: str
    graph: FsmGraph
    current_node: str
    node_statuses: dict[str, NodeStatus]
    state_bag: dict = field(default_factory=dict)
    phase_start_ms: int = 0
    skill_start_ms: int = 0
    outcome: SkillOutcomeState = SkillOutcomeState.IN_PROGRESS
    failure_code: SkillFailureCode | None = None
    transition_history: list[TransitionRecord] = field(default_factory=list)

    @property
    def phase_id(self) -> PhaseId:
        return self.graph.nodes[self.current_node].phase_id

    @property
    def is_active(self) -> bool:
        return self.outcome == SkillOutcomeState.IN_PROGRESS

    @property
    def is_terminal(self) -> bool:
        return self.current_node in self.graph.terminal_nodes

    @property
    def wrist_camera_active(self) -> bool:
        return self.phase_id in WRIST_ACTIVE_PHASES
