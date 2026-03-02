"""JointPositionControlService — monitors joint-space safety and tracks ActInfo.

In teacher mode the server applies actions directly; this service only validates
joint limits and updates ActInfo in the runtime store. It does NOT call apply_fn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from halo.contracts.actions import JointPositionAction, JointPositionChunk
from halo.contracts.enums import WRIST_ACTIVE_PHASES, ActStatus, SafetyState
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import ActInfo, SafetyInfo
from halo.runtime.runtime import HALORuntime
from halo.services.control_service.config import JointControlConfig
from halo.services.control_service.joint_safety_guard import JointSafetyGuard

logger = logging.getLogger(__name__)


class JointPositionControlService:
    """Joint-position monitoring service for teacher-driven control.

    Mirrors ControlService pattern but only monitors — teacher handles
    actual action application on the server side.

    Lifecycle:
        svc = JointPositionControlService(arm_id, runtime)
        await svc.start()
        await svc.push_chunk(chunk)
        await svc.stop()
    """

    def __init__(
        self,
        arm_id: str,
        runtime: HALORuntime,
        config: JointControlConfig = JointControlConfig(),
    ) -> None:
        self._arm_id = arm_id
        self._runtime = runtime
        self._config = config

        self._buffer: deque[JointPositionAction] = deque()
        self._guard = JointSafetyGuard(config)
        self._reflex_active: bool = False

        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._event_task: asyncio.Task | None = None
        self._phase_queue: asyncio.Queue[EventEnvelope] | None = None

    # --- Public API ---

    async def start(self) -> None:
        """Spawn the monitoring loop and event-drain tasks."""
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

    async def push_chunk(self, chunk: JointPositionChunk) -> None:
        """Enqueue a JointPositionChunk into the FIFO buffer."""
        if chunk.arm_id != self._arm_id:
            raise ValueError(f"chunk.arm_id {chunk.arm_id!r} != service arm_id {self._arm_id!r}")
        self._buffer.extend(chunk.actions)

    # --- Testable internal ---

    async def tick(self) -> JointPositionAction | None:
        """One monitoring tick.

        1. Pop action from FIFO
        2. Safety check (joint-limit validation)
        3. If violation: trigger reflex, update SafetyInfo
        4. If clean after reflex: recover
        5. Update ActInfo (status, buffer_fill_ms, wrist_enabled)
        6. No apply (teacher already applied on server)
        """
        # Derive wrist_enabled from current phase
        snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)
        wrist = snap.skill is not None and snap.skill.phase in WRIST_ACTIVE_PHASES

        if not self._buffer:
            await self._runtime.store.update_act(
                self._arm_id,
                ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False, wrist_enabled=wrist),
            )
            return None

        action = self._buffer.popleft()
        fill_ms = int(len(self._buffer) * (1000.0 / self._config.control_rate_hz))
        low = fill_ms < self._config.buffer_low_threshold_ms

        # Safety check
        violations = self._guard.check(action)
        if violations:
            if not self._reflex_active:
                await self._trigger_reflex(violations)
            status = ActStatus.BUFFER_LOW if low else ActStatus.RUNNING
            await self._runtime.store.update_act(
                self._arm_id,
                ActInfo(status=status, buffer_fill_ms=fill_ms, buffer_low=low, wrist_enabled=wrist),
            )
            return None

        # Recover from reflex if clean
        if self._reflex_active:
            await self._recover_from_reflex()

        status = ActStatus.BUFFER_LOW if low else ActStatus.RUNNING
        await self._runtime.store.update_act(
            self._arm_id,
            ActInfo(status=status, buffer_fill_ms=fill_ms, buffer_low=low, wrist_enabled=wrist),
        )
        return action

    # --- Private ---

    async def _run_loop(self) -> None:
        period = 1.0 / self._config.control_rate_hz
        while not self._stop_event.is_set():
            await self.tick()
            await asyncio.sleep(period)

    async def _on_phase_event(self, event: EventEnvelope) -> None:
        """On PHASE_ENTER: clear buffer to discard stale actions."""
        if event.type == EventType.PHASE_ENTER:
            self._buffer.clear()

    async def _trigger_reflex(self, reasons: list) -> None:
        self._reflex_active = True
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
        self._reflex_active = False
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
        self._phase_queue = self._runtime.bus.subscribe(self._arm_id)
        self._event_task = asyncio.create_task(self._drain_events())

    async def _drain_events(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(self._phase_queue.get(), timeout=0.05)
                await self._on_phase_event(event)
            except asyncio.TimeoutError:
                continue
