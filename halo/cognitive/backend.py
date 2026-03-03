"""CognitiveBackend protocol — brain + eyes behind a single interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.target_perception_service.vlm_parser import VlmScene


@runtime_checkable
class CognitiveBackend(Protocol):
    """Protocol for a paired brain (planner LLM) + eyes (VLM) backend."""

    @property
    def backend_type(self) -> str: ...

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
    ) -> list[CommandEnvelope]: ...

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene: ...

    async def health_check(self) -> bool: ...

    @property
    def last_reasoning(self) -> str: ...

    def reset_loop_state(self) -> None: ...
