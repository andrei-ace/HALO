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
from halo.cognitive.compactor import CompactionResult
from halo.cognitive.config import BackendReadiness, BackendType, CognitiveConfig
from halo.cognitive.context_store import ContextStore
from halo.cognitive.lease import LeaseManager
from halo.contracts.commands import CommandEnvelope
from halo.contracts.events import EventType
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.target_perception_service.vlm_parser import VlmScene

if TYPE_CHECKING:
    from halo.cognitive.backend import CognitiveBackend
    from halo.runtime.event_bus import EventBus

logger = logging.getLogger(__name__)

CONSECUTIVE_FAILURES_BEFORE_SWITCH = 3
_CATCHUP_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAYS = (0.5, 1.0, 2.0)

# Errors that should immediately fail over without retrying
_NO_RETRY_KEYWORDS = ("429", "RESOURCE_EXHAUSTED", "quota")


def _is_non_retryable(exc: Exception) -> bool:
    """Return True if the error should skip retries and fail over immediately."""
    msg = str(exc).lower()
    return any(kw.lower() in msg for kw in _NO_RETRY_KEYWORDS)


_JOURNALED_EVENT_TYPES = frozenset(
    {
        EventType.SKILL_STARTED,
        EventType.SKILL_SUCCEEDED,
        EventType.SKILL_FAILED,
        EventType.SAFETY_REFLEX_TRIGGERED,
        EventType.TARGET_ACQUIRED,
        EventType.PERCEPTION_FAILURE,
    }
)


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
        arm_id: str = "arm0",
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS,
    ) -> None:
        self._config = config
        self._backends: dict[str, CognitiveBackend] = {
            BackendType.LOCAL: local,
            BackendType.CLOUD: cloud,
        }
        self._lease_mgr = lease_mgr if lease_mgr is not None else LeaseManager()
        self._context_store = context_store if context_store is not None else ContextStore()
        self._bus = bus
        self._snapshot_fn = snapshot_fn
        self._arm_id = arm_id

        # Retry config
        self._max_retries = max_retries
        self._retry_delays = retry_delays

        # Health tracking
        self._consecutive_failures: int = 0
        self._last_failure_reason: str = ""
        self._health_task: asyncio.Task | None = None
        self._journal_task: asyncio.Task | None = None

        # Grant initial lease to the configured active backend
        self._active_type: BackendType = config.active
        self._lease_mgr.grant(self._active_type)

        # Wire compaction callback on cloud backend (if it supports it)
        if hasattr(cloud, "set_on_compaction"):
            cloud.set_on_compaction(self._sync_compaction_to_inactive)

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
        """Delegate to active backend.decide() with retry on transient errors."""
        # Record operator instruction eagerly so it survives failover
        if operator_cmd:
            self._context_store.append(
                epoch=self._lease_mgr.current_epoch,
                backend=self._active_type,
                entry_type="operator",
                summary=operator_cmd,
            )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                commands = await self.active_backend.decide(
                    snap,
                    operator_cmd=operator_cmd,
                    epoch=self._lease_mgr.current_epoch,
                )

                # Detect empty responses (no reasoning, no commands)
                reasoning = self.active_backend.last_reasoning
                if not commands and not reasoning:
                    logger.warning("decide() empty response on %s", self._active_type)
                    await self._on_failure(reason="empty response (no reasoning, no commands)")
                    return []

                self._on_success()

                # Stamp epoch + lease_token on all returned commands
                from dataclasses import replace as _dc_replace

                epoch = self._lease_mgr.current_epoch
                token = self._lease_mgr.current_token
                if token is not None:
                    commands = [_dc_replace(c, epoch=epoch, lease_token=token) for c in commands]

                # Record decision in context store (reasoning already fetched above)
                if reasoning:
                    self._context_store.append(
                        epoch=self._lease_mgr.current_epoch,
                        backend=self._active_type,
                        entry_type="decision",
                        summary=reasoning,
                    )

                return commands
            except Exception as exc:
                last_exc = exc
                if _is_non_retryable(exc):
                    logger.info("decide() non-retryable error on %s: %s", self._active_type, exc)
                    break
                delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                logger.info(
                    "decide() attempt %d/%d on %s: %s — retry in %.0fs",
                    attempt + 1,
                    self._max_retries,
                    self._active_type,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        else:
            logger.warning("decide() failed after %d retries on %s: %s", self._max_retries, self._active_type, last_exc)
        # Force immediate failover — retries exhausted or non-retryable
        old_type = self._active_type
        self._consecutive_failures = CONSECUTIVE_FAILURES_BEFORE_SWITCH - 1
        await self._on_failure(reason=str(last_exc) if last_exc else "unknown error")

        # If we switched, replay the call on the new backend
        if self._active_type != old_type:
            try:
                commands = await self.active_backend.decide(
                    snap,
                    operator_cmd=operator_cmd,
                    epoch=self._lease_mgr.current_epoch,
                )
                self._on_success()
                from dataclasses import replace as _dc_replace

                epoch = self._lease_mgr.current_epoch
                token = self._lease_mgr.current_token
                if token is not None:
                    commands = [_dc_replace(c, epoch=epoch, lease_token=token) for c in commands]
                reasoning = self.active_backend.last_reasoning
                if reasoning:
                    self._context_store.append(
                        epoch=self._lease_mgr.current_epoch,
                        backend=self._active_type,
                        entry_type="decision",
                        summary=reasoning,
                    )
                return commands
            except Exception as replay_exc:
                logger.warning("decide() replay on %s also failed: %s", self._active_type, replay_exc)
        return []

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        """Delegate to active backend.vlm_scene() with retry on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
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
            except Exception as exc:
                last_exc = exc
                if _is_non_retryable(exc):
                    logger.info("vlm_scene() non-retryable error on %s: %s", self._active_type, exc)
                    break
                delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                logger.info(
                    "vlm_scene() attempt %d/%d on %s: %s — retry in %.0fs",
                    attempt + 1,
                    self._max_retries,
                    self._active_type,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        else:
            logger.warning(
                "vlm_scene() failed after %d retries on %s: %s", self._max_retries, self._active_type, last_exc
            )
        # Force immediate failover — retries exhausted or non-retryable
        old_type = self._active_type
        self._consecutive_failures = CONSECUTIVE_FAILURES_BEFORE_SWITCH - 1
        await self._on_failure(reason=str(last_exc) if last_exc else "unknown error")

        # If we switched, replay the call on the new backend
        if self._active_type != old_type:
            try:
                scene = await self.active_backend.vlm_scene(arm_id, image, known_handles, target_handle=target_handle)
                self._on_success()
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
            except Exception as replay_exc:
                logger.warning("vlm_scene() replay on %s also failed: %s", self._active_type, replay_exc)
        return VlmScene(scene="", detections=[])

    @property
    def model_name(self) -> str:
        return getattr(self.active_backend, "model_name", "")

    @property
    def last_reasoning(self) -> str:
        return self.active_backend.last_reasoning

    @property
    def last_token_usage(self) -> dict[str, int]:
        return getattr(self.active_backend, "last_token_usage", {})

    def reset_loop_state(self) -> None:
        self.active_backend.reset_loop_state()

    # ------------------------------------------------------------------
    # Switching
    # ------------------------------------------------------------------

    async def switch_to(self, target: BackendType, reason: str = "") -> None:
        """Switch to *target* backend with context handoff.

        Protocol:
        1. Snapshot context from current backend
        2. Pre-warm new backend (best-effort, while old lease still valid)
        3. Revoke old lease
        4. Grant new lease to target backend
        5. Reset loop state on new backend
        6. Publish BACKEND_SWITCHED event (if bus available)
        """
        if target == self._active_type:
            logger.info("Already on %s backend, skipping switch", target)
            return

        old_type = self._active_type
        old_epoch = self._lease_mgr.current_epoch

        logger.info("Switching backend: %s -> %s (reason: %s)", old_type, target, reason)

        # 1. Snapshot context
        _snapshot = self._context_store.take_snapshot(old_epoch)

        # 2. Pre-warm new backend (best-effort, old lease still active)
        new_backend = self._backends[target]
        if isinstance(new_backend, WarmableBackend):
            try:
                snapshot = None
                if self._snapshot_fn is not None:
                    try:
                        snapshot = await self._snapshot_fn(self._arm_id)
                    except Exception:
                        pass
                state = self._context_store.build_cognitive_state(
                    epoch=old_epoch,
                    snapshot=snapshot,
                )
                entries = self._context_store.get_entries_after(-1)
                await new_backend.warm_up(state=state, journal_entries=entries)
            except Exception:
                logger.warning("Pre-warm failed for %s during switch", target)

        # 3. Revoke old lease
        self._lease_mgr.revoke(old_epoch)

        # 4. Grant new lease
        self._active_type = target
        self._lease_mgr.grant(target)
        self._consecutive_failures = 0

        # 5. Reset loop state on old backend only (drain stale commands / mark
        #    COLD for future failback).  The new backend was just warmed up in
        #    step 2 — resetting it would undo that warm-up and force a redundant
        #    re-warm on the next health tick.
        self._backends[old_type].reset_loop_state()

        # 6. Publish event + log to run logger
        switch_data = {
            "from": old_type,
            "to": target,
            "reason": reason,
            "epoch": self._lease_mgr.current_epoch,
        }
        if self._bus is not None:
            from halo.contracts.events import EventEnvelope, EventType

            event = EventEnvelope(
                event_id=f"switch-{self._lease_mgr.current_epoch}",
                type=EventType.BACKEND_SWITCHED,
                ts_ms=int(time.time() * 1000),
                arm_id=self._arm_id,
                data=switch_data,
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
        """Start background health monitoring (if failover enabled) and event journal."""
        if self._config.enable_failover and self._health_task is None:
            self._health_task = asyncio.create_task(self._health_loop())
        if self._bus is not None and self._journal_task is None:
            self._journal_task = asyncio.create_task(self._event_journal_loop())

    async def stop(self) -> None:
        """Stop health monitoring and event journal."""
        for task in (self._health_task, self._journal_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._health_task = None
        self._journal_task = None

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
                        await self._on_failure(reason="health check failed")
                except Exception as exc:
                    await self._on_failure(reason=f"health check error: {exc}")
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
                    snapshot = await self._snapshot_fn(self._arm_id)
                except Exception:
                    pass
            state = self._context_store.build_cognitive_state(
                epoch=self._lease_mgr.current_epoch,
                snapshot=snapshot,
            )
            entries = self._context_store.get_entries_after(-1)
            await backend.warm_up(state=state, journal_entries=entries)
        except Exception:
            logger.debug("Re-warm of active backend failed", exc_info=True)

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
                        snapshot = await self._snapshot_fn(self._arm_id)
                    except Exception:
                        pass
                state = self._context_store.build_cognitive_state(
                    epoch=self._lease_mgr.current_epoch,
                    snapshot=snapshot,
                )
                entries = self._context_store.get_entries_after(-1)
                await preferred_backend.warm_up(state=state, journal_entries=entries)

            elif readiness == BackendReadiness.WARMING:
                # Incremental catch-up (bounded batch)
                entries = self._context_store.get_entries_after(preferred_backend.caught_up_cursor)
                await preferred_backend.warm_up(state=None, journal_entries=entries[:_CATCHUP_BATCH_SIZE])

            elif readiness == BackendReadiness.READY:
                # Verify cursor parity before switching
                backend_cursor = preferred_backend.caught_up_cursor
                store_cursor = self._context_store.latest_cursor
                if backend_cursor < store_cursor:
                    logger.info(
                        "Preferred backend READY but cursor behind (%d < %d), sending catchup",
                        backend_cursor,
                        store_cursor,
                    )
                    entries = self._context_store.get_entries_after(backend_cursor)
                    if not entries:
                        entries = self._context_store.get_entries_after(-1)
                    await preferred_backend.warm_up(state=None, journal_entries=entries[:_CATCHUP_BATCH_SIZE])
                    return
                # Caught up (backend_cursor >= store_cursor) — switch
                await self.switch_to(preferred, reason="preferred backend recovered and warmed up")

        except Exception:
            pass  # Preferred still unhealthy or warm-up failed

    async def _event_journal_loop(self) -> None:
        """Subscribe to EventBus and journal key runtime events."""
        if self._bus is None:
            return
        queue = self._bus.subscribe(self._arm_id, maxsize=0)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    continue
                if event.type not in _JOURNALED_EVENT_TYPES:
                    continue
                summary = event.type.value
                data = dict(event.data) if event.data else {}
                details = {
                    k: v
                    for k, v in data.items()
                    if k in ("target_handle", "failure_code", "reason", "phase", "skill_name", "reflex_reason")
                }
                if details:
                    summary += f": {details}"
                self._context_store.append(
                    epoch=self._lease_mgr.current_epoch,
                    backend=self._active_type,
                    entry_type="event",
                    summary=summary,
                    data={"event_type": event.type.value, **data},
                )
        except asyncio.CancelledError:
            if self._bus is not None:
                self._bus.unsubscribe(self._arm_id, queue)

    async def _sync_compaction_to_inactive(self, result: CompactionResult) -> None:
        """Propagate compaction summary + retained messages to the inactive backend.

        After ADK compacts the cloud session, this rebuilds the local backend's
        session with the compaction summary and retained conversation history so
        that on failover the local model starts with proper context.
        """
        inactive_type = BackendType.LOCAL if self._active_type == BackendType.CLOUD else BackendType.CLOUD
        inactive = self._backends.get(inactive_type)
        if inactive is None:
            return

        try:
            from halo.cognitive.local_backend import LocalCognitiveBackend

            if isinstance(inactive, LocalCognitiveBackend):
                # Extract retained records from active backend's MessageHistory
                active = self._backends[self._active_type]
                retained = []
                if hasattr(active, "agent") and hasattr(active.agent, "msg_history"):
                    retained = [r for r in active.agent.msg_history.get_all() if not r.is_summary]

                await inactive.agent.inject_compaction_state(result.summary, retained)
        except Exception:
            logger.debug("Failed to sync compaction to inactive backend", exc_info=True)

        # Record compaction in context store
        self._context_store.append(
            epoch=self._lease_mgr.current_epoch,
            backend=self._active_type,
            entry_type="compaction",
            summary=f"Session compacted: {result.compacted_count} messages → summary",
            data={"up_to_msg_id": result.up_to_msg_id, "retained_count": result.retained_count},
        )

        # Publish SESSION_COMPACTED event (filtered from planner history, like BACKEND_SWITCHED)
        if self._bus is not None:
            from halo.contracts.events import EventEnvelope, EventType

            event = EventEnvelope(
                event_id=f"compaction-{self._lease_mgr.current_epoch}-{result.up_to_msg_id[:8]}",
                type=EventType.SESSION_COMPACTED,
                ts_ms=result.ts_ms,
                arm_id=self._arm_id,
                data={
                    "compacted_count": result.compacted_count,
                    "retained_count": result.retained_count,
                    "backend": self._active_type,
                    "summary": result.summary,
                    "up_to_msg_id": result.up_to_msg_id,
                },
            )
            await self._bus.publish(event)

    def _on_success(self) -> None:
        """Reset consecutive failure counter and renew lease TTL on success."""
        self._consecutive_failures = 0
        self._last_failure_reason = ""
        self._lease_mgr.renew(self._lease_mgr.current_epoch)

    async def _on_failure(self, reason: str = "") -> None:
        """Increment failure counter and switch if threshold reached."""
        if reason:
            self._last_failure_reason = reason
        self._consecutive_failures += 1
        if self._config.enable_failover and self._consecutive_failures >= CONSECUTIVE_FAILURES_BEFORE_SWITCH:
            if self._active_type == BackendType.CLOUD:
                target = BackendType.LOCAL
            else:
                target = BackendType.CLOUD
            switch_reason = f"{self._consecutive_failures} consecutive failures: {self._last_failure_reason}"
            await self.switch_to(target, reason=switch_reason)
