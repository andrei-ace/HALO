from __future__ import annotations

import asyncio
import time
import uuid
from typing import Awaitable, Callable

from halo.contracts.commands import CommandAck, CommandEnvelope, DescribeScenePayload
from halo.contracts.enums import CommandType
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import PlannerSnapshot
from halo.runtime.runtime import HALORuntime
from halo.services.planner_service.config import PlannerServiceConfig

DecideFn = Callable[[PlannerSnapshot], Awaitable[list[CommandEnvelope]]]

_URGENT_EVENT_TYPES = frozenset({
    EventType.SKILL_SUCCEEDED,
    EventType.SKILL_FAILED,
    EventType.SAFETY_REFLEX_TRIGGERED,
    EventType.PERCEPTION_FAILURE,
    EventType.SCENE_DESCRIBED,
    EventType.TARGET_ACQUIRED,
})


class PlannerService:
    """
    Event-driven LLM orchestration loop. Reads snapshots, calls decide_fn,
    and submits commands back to the runtime.

    A tick fires when an urgent event arrives (SKILL_SUCCEEDED, SKILL_FAILED,
    SAFETY_REFLEX_TRIGGERED, PERCEPTION_FAILURE) or when the watchdog timeout
    expires with no event. decide_fn is awaited before the next event is
    processed, so ticks are never concurrent regardless of LLM latency.

    Lifecycle:
        svc = PlannerService(arm_id, runtime, decide_fn)
        await svc.start()
        await svc.stop()

    decide_fn is the only coupling to the LLM:
        async def decide_fn(snap: PlannerSnapshot) -> list[CommandEnvelope]: ...
    """

    def __init__(
        self,
        arm_id: str,
        runtime: HALORuntime,
        decide_fn: DecideFn,
        config: PlannerServiceConfig = PlannerServiceConfig(),
    ) -> None:
        self._arm_id = arm_id
        self._runtime = runtime
        self._decide_fn = decide_fn
        self._config = config

        self._stop_event = asyncio.Event()
        self._urgent_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._event_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue | None = None

    # --- Public API ---

    async def start(self) -> None:
        """Spawn the planner loop and event-drain tasks.

        Immediately issues a DESCRIBE_SCENE so the VLM acquires the
        initial scene before the first planner tick fires.
        """
        self._stop_event.clear()
        self._urgent_event.clear()
        self._event_queue = self._runtime.bus.subscribe(self._arm_id)
        self._event_task = asyncio.create_task(self._drain_events())
        self._loop_task = asyncio.create_task(self._run_loop())
        await self._runtime.submit_command(
            CommandEnvelope(
                command_id=str(uuid.uuid4()),
                arm_id=self._arm_id,
                issued_at_ms=int(time.time() * 1000),
                type=CommandType.DESCRIBE_SCENE,
                payload=DescribeScenePayload(
                    reason="planner startup — initial VLM scene acquisition",
                ),
                precondition_snapshot_id=None,
            )
        )

    async def stop(self) -> None:
        """Signal the loop to stop and await clean shutdown."""
        self._stop_event.set()
        self._urgent_event.set()  # unblock wait_for so the loop sees stop_event
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
        if self._event_queue is not None:
            self._runtime.bus.unsubscribe(self._arm_id, self._event_queue)
            self._event_queue = None

    # --- Testable internal ---

    async def tick(self) -> list[CommandAck]:
        """
        One planner tick. Callable directly in tests.
        1. Get latest snapshot
        2. Call decide_fn
        3. Submit up to max_commands_per_tick commands
        4. Return list of CommandAck
        """
        snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)
        commands = await self._decide_fn(snap)
        acks: list[CommandAck] = []
        for cmd in commands[: self._config.max_commands_per_tick]:
            ack = await self._runtime.submit_command(cmd)
            acks.append(ack)
        return acks

    # --- Private ---

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            # Wait for an urgent event or the watchdog timeout.
            # Ticks only happen in response to events (or the watchdog),
            # so LLM latency never causes tick pile-up.
            try:
                await asyncio.wait_for(
                    self._urgent_event.wait(),
                    timeout=self._config.watchdog_interval_s,
                )
                self._urgent_event.clear()
            except asyncio.TimeoutError:
                pass  # watchdog fired — tick anyway as a safety net

            if self._stop_event.is_set():
                break

            await self.tick()

    async def _drain_events(self) -> None:
        while not self._stop_event.is_set():
            try:
                event: EventEnvelope = await asyncio.wait_for(
                    self._event_queue.get(), timeout=0.05
                )
                if event.type in _URGENT_EVENT_TYPES:
                    self._urgent_event.set()
            except asyncio.TimeoutError:
                continue
