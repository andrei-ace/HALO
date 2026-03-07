from __future__ import annotations

from dataclasses import dataclass

from halo.services.skill_runner_service.graph import FsmEdge
from halo.services.skill_runner_service.queue import SkillQueue
from halo.services.skill_runner_service.skill_run import SkillRun, TransitionRecord


@dataclass(frozen=True)
class NodeViewModel:
    name: str
    phase_id: int
    status: str
    elapsed_ms: int | None


@dataclass(frozen=True)
class QueuedSkillViewModel:
    skill_name: str
    variant: str
    target_handle: str
    position: int


@dataclass(frozen=True)
class FsmViewModel:
    skill_name: str
    variant: str
    skill_run_id: str
    target_handle: str
    nodes: tuple[NodeViewModel, ...]
    edges: tuple[FsmEdge, ...]
    current_node: str
    outcome: str
    failure_code: str | None
    transition_history: tuple[TransitionRecord, ...]
    mermaid_source: str
    queued_skills: tuple[QueuedSkillViewModel, ...]


def build_fsm_view_model(
    active_run: SkillRun | None,
    queue: SkillQueue,
    now_ms: int,
) -> FsmViewModel | None:
    if active_run is None:
        return None

    nodes: list[NodeViewModel] = []
    for name, node in active_run.graph.nodes.items():
        status = active_run.node_statuses.get(name, "PENDING")
        elapsed = now_ms - active_run.phase_start_ms if name == active_run.current_node else None
        nodes.append(NodeViewModel(name=name, phase_id=int(node.phase_id), status=str(status), elapsed_ms=elapsed))

    queued: list[QueuedSkillViewModel] = []
    for i, item in enumerate(queue.items):
        queued.append(
            QueuedSkillViewModel(
                skill_name=str(item.skill_name),
                variant=item.variant,
                target_handle=item.target_handle,
                position=i,
            )
        )

    return FsmViewModel(
        skill_name=str(active_run.skill_name),
        variant=active_run.variant,
        skill_run_id=active_run.skill_run_id,
        target_handle=active_run.target_handle,
        nodes=tuple(nodes),
        edges=active_run.graph.edges,
        current_node=active_run.current_node,
        outcome=str(active_run.outcome),
        failure_code=active_run.failure_code.value if active_run.failure_code else None,
        transition_history=tuple(active_run.transition_history),
        mermaid_source=active_run.graph.mermaid_source,
        queued_skills=tuple(queued),
    )
