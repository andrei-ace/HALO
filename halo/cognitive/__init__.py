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
        audio_capture: Optional AudioCapture for cloud backend voice input.
        audio_playback: Optional AudioPlayback for cloud backend voice output.
        lease_mgr: Optional LeaseManager — when provided, shared with HALORuntime.
        cloud_backend: Optional pre-built cloud backend (e.g. RemoteCognitiveBackend).
        arm_id: Arm ID for Switchboard event subscriptions. Defaults to "arm0".
    """
    from halo.cognitive.cloud_backend import CloudCognitiveBackend
    from halo.cognitive.local_backend import LocalCognitiveBackend

    cfg = config or CognitiveConfig()
    context_store = ContextStore()
    lease_mgr = lease_mgr or LeaseManager()

    local = LocalCognitiveBackend(config=cfg.local, run_logger=run_logger)

    if cloud_backend is not None:
        cloud = cloud_backend
    else:
        cloud = CloudCognitiveBackend(
            config=cfg.cloud,
            audio_capture=audio_capture,
            audio_playback=audio_playback,
            run_logger=run_logger,
        )

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
