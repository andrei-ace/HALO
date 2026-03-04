"""Cognitive backend abstraction — brain (planner) + eyes (VLM) switching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

from halo.cognitive.backend import CognitiveBackend
from halo.cognitive.config import BackendType, CloudConfig, CognitiveConfig, LocalConfig
from halo.cognitive.context_store import ContextStore
from halo.cognitive.lease import LeaseManager
from halo.cognitive.switchboard import Switchboard

if TYPE_CHECKING:
    from halo.contracts.snapshots import PlannerSnapshot
    from halo.runtime.event_bus import EventBus


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
) -> CognitiveStack:
    """Factory that wires up all cognitive components from config.

    Args:
        config: Cognitive configuration. Defaults to CognitiveConfig().
        bus: Optional EventBus for BACKEND_SWITCHED events.
        snapshot_fn: Optional async callable to get latest PlannerSnapshot for warm-up.
        run_logger: Optional RunLogger for local backend VLM logging.
    """
    from halo.cognitive.cloud_backend import CloudCognitiveBackend
    from halo.cognitive.local_backend import LocalCognitiveBackend

    cfg = config or CognitiveConfig()
    context_store = ContextStore()
    lease_mgr = LeaseManager()

    local = LocalCognitiveBackend(config=cfg.local, run_logger=run_logger)
    cloud = CloudCognitiveBackend(config=cfg.cloud)

    switchboard = Switchboard(
        config=cfg,
        local=local,
        cloud=cloud,
        lease_mgr=lease_mgr,
        context_store=context_store,
        bus=bus,
        snapshot_fn=snapshot_fn,
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
