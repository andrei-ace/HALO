"""Cognitive backend abstraction — brain (planner) + eyes (VLM) switching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

from halo.cognitive.backend import CognitiveBackend
from halo.cognitive.config import BackendType, CloudConfig, CognitiveConfig, LocalConfig
from halo.cognitive.context_store import ContextStore
from halo.cognitive.lease import LeaseManager
from halo.cognitive.switchboard import Switchboard
from halo.services.target_perception_service.vlm_parser import VlmScene

if TYPE_CHECKING:
    from halo.contracts.commands import CommandEnvelope
    from halo.contracts.snapshots import PlannerSnapshot
    from halo.runtime.event_bus import EventBus


class _UnavailableCloudBackend:
    """Placeholder cloud backend that keeps LOCAL as the safe default."""

    def __init__(self, reason: str = "Cloud backend unavailable") -> None:
        self._reason = reason

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        raise RuntimeError(self._reason)

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        raise RuntimeError(self._reason)

    async def health_check(self) -> bool:
        return False

    @property
    def last_reasoning(self) -> str:
        return ""

    def reset_loop_state(self) -> None:
        return None


@dataclass
class CognitiveStack:
    """All cognitive components wired together."""

    switchboard: Switchboard
    local: CognitiveBackend
    cloud: CognitiveBackend
    context_store: ContextStore
    lease_manager: LeaseManager
    config: CognitiveConfig


def make_cognitive_stack(
    config: CognitiveConfig | None = None,
    bus: EventBus | None = None,
    snapshot_fn: Callable[[str], Awaitable[PlannerSnapshot | None]] | None = None,
    run_logger: object | None = None,
    audio_capture: object | None = None,
    audio_playback: object | None = None,
    lease_mgr: LeaseManager | None = None,
    cloud_backend: CognitiveBackend | None = None,
    arm_id: str = "arm0",
) -> CognitiveStack:
    """Factory that wires up all cognitive components from config.

    Args:
        config: Cognitive configuration. Defaults to CognitiveConfig().
        bus: Optional EventBus for BACKEND_SWITCHED events.
        snapshot_fn: Optional async callable to get latest PlannerSnapshot for warm-up.
        run_logger: Optional RunLogger for backend VLM logging.
        lease_mgr: Optional LeaseManager — when provided, shared with HALORuntime.
        cloud_backend: Optional pre-built cloud backend (e.g. RemoteCognitiveBackend).
        arm_id: Arm ID for Switchboard event subscriptions. Defaults to "arm0".
    """
    from halo.cognitive.local_backend import LocalCognitiveBackend

    cfg = config or CognitiveConfig()
    context_store = ContextStore()
    lease_mgr = lease_mgr or LeaseManager()

    local = LocalCognitiveBackend(config=cfg.local, run_logger=run_logger)

    if cloud_backend is not None:
        cloud = cloud_backend
    elif cfg.active == BackendType.CLOUD:
        msg = "CognitiveConfig(active=CLOUD) requires cloud_backend to be provided"
        raise ValueError(msg)
    else:
        cloud = _UnavailableCloudBackend()

    switchboard = Switchboard(
        config=cfg,
        local=local,
        cloud=cloud,
        lease_mgr=lease_mgr,
        context_store=context_store,
        bus=bus,
        snapshot_fn=snapshot_fn,
        arm_id=arm_id,
    )

    return CognitiveStack(
        switchboard=switchboard,
        local=local,
        cloud=cloud,
        context_store=context_store,
        lease_manager=lease_mgr,
        config=cfg,
    )


__all__ = [
    "BackendType",
    "CloudConfig",
    "CognitiveBackend",
    "CognitiveConfig",
    "CognitiveStack",
    "LocalConfig",
    "make_cognitive_stack",
]
