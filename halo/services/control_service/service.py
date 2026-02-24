from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from halo.contracts.actions import ZERO_ACTION, Action, ActionChunk
from halo.contracts.enums import ActStatus, PhaseId, SafetyReflexReason, SafetyState
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import ActInfo, SafetyInfo
from halo.runtime.runtime import HALORuntime
from halo.services.control_service.config import ControlServiceConfig
from halo.services.control_service.safety_guard import SafetyGuard
from halo.services.control_service.te_buffer import TemporalEnsemblingBuffer

ApplyFn = Callable[[str, Action], Awaitable[None]]


class ControlService:
    """
    High-rate asyncio loop (default 50 Hz). Owns action buffer, safety
    guard, and reflex state. Never blocks on LLM/VLM.

    Lifecycle:
        svc = ControlService(arm_id, runtime, apply_fn)
        await svc.start()          # spawns background task
        await svc.push_chunk(chunk)
        await svc.stop()

    The apply_fn is the only coupling to the robot/sim:
        async def apply_fn(arm_id: str, action: Action) -> None: ...
    """

    def __init__(
        self,
        arm_id: str,
        runtime: HALORuntime,
        apply_fn: ApplyFn,
        config: ControlServiceConfig = ControlServiceConfig(),
    ) -> None:
        self._arm_id = arm_id
        self._runtime = runtime
        self._apply_fn = apply_fn
        self._config = config

        self._buffer = TemporalEnsemblingBuffer(temp=config.ensembling_temp)
        self._guard = SafetyGuard(config)
        self._lock = asyncio.Lock()

        self._reflex_active: bool = False
        self._reflex_reasons: list[SafetyReflexReason] = []
        self._last_phase: PhaseId | None = None

        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._event_task: asyncio.Task | None = None
        self._phase_queue: asyncio.Queue[EventEnvelope] | None = None

    # --- Public API ---

    async def start(self) -> None:
        """Spawn the control loop and event-drain tasks."""
        self._stop_event.clear()
        await self._subscribe_phase_events()
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop and await clean shutdown."""
        self._stop_event.set()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None
        if self._event_task is not None:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
            self._event_task = None
        if self._phase_queue is not None:
            self._runtime.bus.unsubscribe(self._arm_id, self._phase_queue)
            self._phase_queue = None

    async def push_chunk(self, chunk: ActionChunk) -> None:
        """Enqueue an ActionChunk. Raises ValueError if arm_id mismatches."""
        if chunk.arm_id != self._arm_id:
            raise ValueError(f"chunk.arm_id {chunk.arm_id!r} != service arm_id {self._arm_id!r}")
        async with self._lock:
            self._buffer.push_chunk(chunk)

    # --- Testable internal ---

    async def tick(self) -> Action | None:
        """
        One control tick. Callable directly in tests.
        1. Check hint freshness → hold if stale
        2. Pop action from buffer
        3. SafetyGuard.check() → trigger reflex on violation
        4. SafetyGuard.clamp()
        5. apply_fn(arm_id, clamped)
        6. Update store: ActInfo (status, buffer_fill_ms, buffer_low)
        7. Return applied action (or None if held/empty)
        """
        # 1. Check hint freshness
        snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)
        target = snap.target

        if not self._guard.check_hint_freshness(target, self._config):
            # Stale hint — hold position, no reflex
            await self._apply_fn(self._arm_id, ZERO_ACTION)
            async with self._lock:
                fill = self._buffer.fill_ms(self._config.control_rate_hz)
                low = self._buffer.is_low(self._config.buffer_low_threshold_ms, self._config.control_rate_hz)
            await self._runtime.store.update_act(
                self._arm_id,
                ActInfo(status=ActStatus.STALE, buffer_fill_ms=fill, buffer_low=low),
            )
            return None

        # 2. Pop action from buffer
        async with self._lock:
            action = self._buffer.pop_action()
            fill = self._buffer.fill_ms(self._config.control_rate_hz)
            low = self._buffer.is_low(self._config.buffer_low_threshold_ms, self._config.control_rate_hz)

        if action is None:
            await self._runtime.store.update_act(
                self._arm_id,
                ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
            )
            return None

        # 3. Safety check
        violations = self._guard.check(action)
        if violations:
            if not self._reflex_active:
                await self._trigger_reflex(violations)
            await self._apply_fn(self._arm_id, ZERO_ACTION)
            await self._runtime.store.update_act(
                self._arm_id,
                ActInfo(
                    status=ActStatus.BUFFER_LOW if low else ActStatus.RUNNING,
                    buffer_fill_ms=fill,
                    buffer_low=low,
                ),
            )
            return None

        # Violations cleared — recover from reflex if it was active
        if self._reflex_active:
            await self._recover_from_reflex()

        # 4. Clamp
        clamped = self._guard.clamp(action)

        # 5. Apply
        await self._apply_fn(self._arm_id, clamped)

        # 6. Update store
        status = ActStatus.BUFFER_LOW if low else ActStatus.RUNNING
        await self._runtime.store.update_act(
            self._arm_id,
            ActInfo(status=status, buffer_fill_ms=fill, buffer_low=low),
        )

        return clamped

    # --- Private ---

    async def _run_loop(self) -> None:
        period = 1.0 / self._config.control_rate_hz
        while not self._stop_event.is_set():
            await self.tick()
            await asyncio.sleep(period)

    async def _on_phase_event(self, event: EventEnvelope) -> None:
        """On PHASE_ENTER: trim buffer to config.buffer_trim_ms."""
        if event.type == EventType.PHASE_ENTER:
            async with self._lock:
                self._buffer.trim_to_ms(self._config.buffer_trim_ms, self._config.control_rate_hz)

    async def _trigger_reflex(self, reasons: list[SafetyReflexReason]) -> None:
        """Set safety FAULT, publish SAFETY_REFLEX_TRIGGERED, update store."""
        self._reflex_active = True
        self._reflex_reasons = list(reasons)
        event = EventEnvelope(
            event_id=self._runtime.bus.make_event_id(),
            type=EventType.SAFETY_REFLEX_TRIGGERED,
            ts_ms=int(time.time() * 1000),
            arm_id=self._arm_id,
            data={"reasons": [r.value for r in reasons]},
        )
        await self._runtime.bus.publish(event)
        await self._runtime.store.update_safety(
            self._arm_id,
            SafetyInfo(
                state=SafetyState.FAULT,
                reflex_active=True,
                reason_codes=tuple(reasons),
            ),
        )

    async def _recover_from_reflex(self) -> None:
        """Set safety OK, publish SAFETY_RECOVERED, update store."""
        self._reflex_active = False
        self._reflex_reasons = []
        event = EventEnvelope(
            event_id=self._runtime.bus.make_event_id(),
            type=EventType.SAFETY_RECOVERED,
            ts_ms=int(time.time() * 1000),
            arm_id=self._arm_id,
            data={},
        )
        await self._runtime.bus.publish(event)
        await self._runtime.store.update_safety(
            self._arm_id,
            SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=()),
        )

    async def _subscribe_phase_events(self) -> None:
        """Subscribe to bus and drain queue in a background task."""
        self._phase_queue = self._runtime.bus.subscribe(self._arm_id)
        self._event_task = asyncio.create_task(self._drain_events())

    async def _drain_events(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(self._phase_queue.get(), timeout=0.05)
                await self._on_phase_event(event)
            except asyncio.TimeoutError:
                continue
