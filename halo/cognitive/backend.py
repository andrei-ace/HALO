"""CognitiveBackend protocol — brain + eyes behind a single interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from halo.cognitive.context_store import CognitiveState, ContextEntry
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
        epoch: int | None = None,
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


@runtime_checkable
class WarmableBackend(Protocol):
    """Optional extension for backends that support warm-up before switching.

    Separate from CognitiveBackend to keep the base protocol stable.
    Switchboard checks ``isinstance(backend, WarmableBackend)`` before
    using warm-up methods; non-warmable backends switch immediately.
    """

    async def warm_up(
        self,
        state: CognitiveState | None,
        journal_entries: list[ContextEntry],
    ) -> bool:
        """Send state + journal for catch-up. Returns True when ready."""
        ...

    @property
    def readiness(self) -> str:
        """One of BackendReadiness values."""
        ...

    @property
    def caught_up_cursor(self) -> int:
        """Highest journal cursor processed, or -1 if cold."""
        ...
