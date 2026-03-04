"""Switchboard — transparent brain+eyes proxy with health monitoring and failover.

Delegates ``decide()`` and ``vlm_scene()`` to the active ``CognitiveBackend``.
Tracks consecutive failures and switches backends automatically when
``enable_failover`` is set.  All switching goes through ``switch_to()`` which
handles context snapshot, lease rotation, and handoff injection.

The Switchboard is transparent to PlannerService and TargetPerceptionService —
they call ``switchboard.decide`` / ``switchboard.vlm_scene`` as drop-in
replacements for the underlying callables.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Awaitable, Callable

from halo.cognitive.backend import WarmableBackend
from halo.cognitive.config import BackendReadiness, BackendType, CognitiveConfig
from halo.cognitive.context_store import ContextStore
from halo.cognitive.lease import LeaseManager
from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.target_perception_service.vlm_parser import VlmScene

if TYPE_CHECKING:
    from halo.cognitive.backend import CognitiveBackend
    from halo.runtime.event_bus import EventBus

logger = logging.getLogger(__name__)

CONSECUTIVE_FAILURES_BEFORE_SWITCH = 3


class Switchboard:
    """Proxy that routes brain+eyes calls to the active backend with failover."""

    def __init__(
        self,
        config: CognitiveConfig,
        local: CognitiveBackend,
        cloud: CognitiveBackend,
        lease_mgr: LeaseManager | None = None,
        context_store: ContextStore | None = None,
        bus: EventBus | None = None,
        snapshot_fn: Callable[[str], Awaitable[PlannerSnapshot | None]] | None = None,
    ) -> None:
        self._config = config
        self._backends: dict[str, CognitiveBackend] = {
            BackendType.LOCAL: local,
            BackendType.CLOUD: cloud,
        }
        self._lease_mgr = lease_mgr or LeaseManager()
        self._context_store = context_store or ContextStore()
        self._bus = bus
        self._snapshot_fn = snapshot_fn

        # Health tracking
        self._consecutive_failures: int = 0
        self._health_task: asyncio.Task | None = None

        # Grant initial lease to the configured active backend
        self._active_type: BackendType = config.active
        self._lease_mgr.grant(self._active_type)

    @property
    def active_backend(self) -> CognitiveBackend:
        return self._backends[self._active_type]

    @property
    def active_type(self) -> BackendType:
        return self._active_type

    @property
    def context_store(self) -> ContextStore:
        return self._context_store

    @property
    def lease_manager(self) -> LeaseManager:
        return self._lease_mgr

    # ------------------------------------------------------------------
    # Delegated API
    # ------------------------------------------------------------------

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
    ) -> list[CommandEnvelope]:
        """Delegate to active backend.decide(). Track failures for failover."""
        try:
            commands = await self.active_backend.decide(
                snap,
                operator_cmd=operator_cmd,
                epoch=self._lease_mgr.current_epoch,
            )
            self._on_success()

            # Record decision in context store
            reasoning = self.active_backend.last_reasoning
            if reasoning:
                self._context_store.append(
                    epoch=self._lease_mgr.current_epoch,
                    backend=self._active_type,
                    entry_type="decision",
                    summary=reasoning,
                )

            # Record operator instruction
            if operator_cmd:
                self._context_store.append(
                    epoch=self._lease_mgr.current_epoch,
                    backend=self._active_type,
                    entry_type="operator",
                    summary=operator_cmd,
                )

            return commands
        except Exception:
            logger.exception("decide() failed on %s backend", self._active_type)
            await self._on_failure()
            return []

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        """Delegate to active backend.vlm_scene(). Track failures for failover."""
        try:
            scene = await self.active_backend.vlm_scene(arm_id, image, known_handles, target_handle=target_handle)
            self._on_success()

            # Record scene in context store
            if scene.scene:
                handles = [d.handle for d in scene.detections]
                self._context_store.append(
                    epoch=self._lease_mgr.current_epoch,
                    backend=self._active_type,
                    entry_type="scene",
                    summary=scene.scene,
                    data={"handles": handles},
                )

            return scene
        except Exception:
            logger.exception("vlm_scene() failed on %s backend", self._active_type)
            await self._on_failure()
            return VlmScene(scene="", detections=[])

    @property
    def last_reasoning(self) -> str:
        return self.active_backend.last_reasoning

    def reset_loop_state(self) -> None:
        self.active_backend.reset_loop_state()

    # ------------------------------------------------------------------
    # Switching
    # ------------------------------------------------------------------

    async def switch_to(self, target: BackendType, reason: str = "") -> None:
        """Switch to *target* backend with context handoff.

        Protocol:
        1. Snapshot context from current backend
        2. Revoke old lease (bumps epoch)
        3. Grant new lease to target backend
        4. Reset loop state on new backend
        5. Publish BACKEND_SWITCHED event (if bus available)
        """
        if target == self._active_type:
            logger.info("Already on %s backend, skipping switch", target)
            return

        old_type = self._active_type
        old_epoch = self._lease_mgr.current_epoch

        logger.info("Switching backend: %s -> %s (reason: %s)", old_type, target, reason)

        # 1. Snapshot context
        _snapshot = self._context_store.take_snapshot(old_epoch)

        # 2. Revoke old lease
        self._lease_mgr.revoke(old_epoch)

        # 3. Grant new lease
        self._active_type = target
        self._lease_mgr.grant(target)
        self._consecutive_failures = 0

        # 4. Reset loop state on new backend
        self._backends[target].reset_loop_state()

        # 5. Publish event
        if self._bus is not None:
            from halo.contracts.events import EventEnvelope, EventType

            event = EventEnvelope(
                event_id=f"switch-{self._lease_mgr.current_epoch}",
                type=EventType.BACKEND_SWITCHED,
                ts_ms=int(time.monotonic() * 1000),
                arm_id="system",
                data={
                    "from": old_type,
                    "to": target,
                    "reason": reason,
                    "epoch": self._lease_mgr.current_epoch,
                },
            )
            await self._bus.publish(event)

        logger.info("Backend switched to %s (epoch %d)", target, self._lease_mgr.current_epoch)

    def get_handoff_context(self) -> str:
        """Get handoff context text for the new backend's first message."""
        return self._context_store.get_handoff_context(self._lease_mgr.current_epoch)

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background health monitoring (if failover enabled)."""
        if self._config.enable_failover and self._health_task is None:
            self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        """Stop health monitoring."""
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    async def _health_loop(self) -> None:
        """Periodically check active backend health."""
        try:
            while True:
                await asyncio.sleep(self._config.health_check_interval_s)
                try:
                    healthy = await self.active_backend.health_check()
                    if healthy:
                        self._on_success()
                        # Re-warm active backend if it dropped to COLD (e.g. instance restart)
                        await self._rewarm_active()
                        # Check if preferred backend is healthy for failback
                        await self._check_failback()
                    else:
                        await self._on_failure()
                except Exception:
                    await self._on_failure()
        except asyncio.CancelledError:
            pass

    async def _rewarm_active(self) -> None:
        """Re-warm the active backend if it dropped to COLD (e.g. cloud instance restart)."""
        backend = self.active_backend
        if not isinstance(backend, WarmableBackend):
            return
        if backend.readiness != BackendReadiness.COLD:
            return

        logger.info("Active backend %s dropped to COLD — re-warming", self._active_type)
        try:
            snapshot = None
            if self._snapshot_fn is not None:
                try:
                    snapshot = await self._snapshot_fn("arm0")
                except Exception:
                    pass
            state = self._context_store.build_cognitive_state(
                epoch=self._lease_mgr.current_epoch,
                snapshot=snapshot,
            )
            entries = self._context_store.get_entries_after(-1)
            await backend.warm_up(state=state, journal_entries=entries)
        except Exception:
            logger.exception("Re-warm of active backend failed")

    async def _check_failback(self) -> None:
        """If we're on the fallback backend and the preferred one is healthy, switch back.

        Uses warm-up protocol for WarmableBackend implementations:
        - COLD/FAILED: send full CognitiveState + journal via warm_up()
        - WARMING: send incremental journal entries via warm_up()
        - READY: switch to preferred backend
        - Non-WarmableBackend: immediate switch (backward compat)
        """
        preferred = self._config.active
        if self._active_type == preferred:
            return

        try:
            preferred_backend = self._backends[preferred]
            healthy = await preferred_backend.health_check()
            if not healthy:
                return

            # Non-warmable backend: immediate switch (backward compat)
            if not isinstance(preferred_backend, WarmableBackend):
                await self.switch_to(preferred, reason="preferred backend recovered")
                return

            readiness = preferred_backend.readiness
            if readiness in (BackendReadiness.COLD, BackendReadiness.FAILED):
                # Full state + journal warm-up
                snapshot = None
                if self._snapshot_fn is not None:
                    try:
                        snapshot = await self._snapshot_fn("arm0")
                    except Exception:
                        pass
                state = self._context_store.build_cognitive_state(
                    epoch=self._lease_mgr.current_epoch,
                    snapshot=snapshot,
                )
                entries = self._context_store.get_entries_after(-1)
                await preferred_backend.warm_up(state=state, journal_entries=entries)

            elif readiness == BackendReadiness.WARMING:
                # Incremental catch-up
                entries = self._context_store.get_entries_after(preferred_backend.caught_up_cursor)
                await preferred_backend.warm_up(state=None, journal_entries=entries)

            elif readiness == BackendReadiness.READY:
                # Caught up — switch
                await self.switch_to(preferred, reason="preferred backend recovered and warmed up")

        except Exception:
            pass  # Preferred still unhealthy or warm-up failed

    def _on_success(self) -> None:
        """Reset consecutive failure counter and renew lease TTL on success."""
        self._consecutive_failures = 0
        self._lease_mgr.renew(self._lease_mgr.current_epoch)

    async def _on_failure(self) -> None:
        """Increment failure counter and switch if threshold reached."""
        self._consecutive_failures += 1
        if self._config.enable_failover and self._consecutive_failures >= CONSECUTIVE_FAILURES_BEFORE_SWITCH:
            # Switch to the other backend
            target = BackendType.CLOUD if self._active_type == BackendType.LOCAL else BackendType.LOCAL
            await self.switch_to(target, reason=f"{self._consecutive_failures} consecutive failures")
