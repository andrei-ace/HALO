from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from halo.contracts.actions import ActionChunk
from halo.contracts.enums import PhaseId, SkillName, SkillOutcomeState
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import OutcomeInfo, ProgressInfo, SkillInfo
from halo.runtime.runtime import HALORuntime
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.fsm import PickFSM

# Injected callables — decoupled from ACT model and ControlService instance
ChunkFn = Callable[
    [str, PhaseId, object],  # (arm_id, phase, PlannerSnapshot)
    Awaitable[ActionChunk | None],
]
PushFn = Callable[[ActionChunk], Awaitable[None]]


class SkillRunnerService:
    """
    10–20 Hz asyncio loop owning the Pick-skill FSM, ACT chunk scheduling,
    and PHASE_ENTER/EXIT/SKILL_* event publishing.

    Lifecycle:
        svc = SkillRunnerService(arm_id, runtime, chunk_fn, push_fn)
        await svc.start()
        await svc.start_skill(skill_name, skill_run_id, target_handle)
        await svc.stop()
    """

    def __init__(
        self,
        arm_id: str,
        runtime: HALORuntime,
        chunk_fn: ChunkFn,
        push_fn: PushFn,
        config: SkillRunnerConfig = SkillRunnerConfig(),
    ) -> None:
        self._arm_id = arm_id
        self._runtime = runtime
        self._chunk_fn = chunk_fn
        self._push_fn = push_fn
        self._config = config

        self._fsm: PickFSM = PickFSM(config)
        self._skill_name: SkillName | None = None
        self._skill_run_id: str | None = None
        self._skill_start_ms: int = 0
        self._last_distance_m: float | None = None

        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None

    # --- Public API ---

    async def start(self) -> None:
        """Spawn the runner loop."""
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop and await clean shutdown."""
        self._stop_event.set()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None

    async def start_skill(
        self,
        skill_name: SkillName,
        skill_run_id: str,
        target_handle: str,
    ) -> None:
        """Begin a new Pick skill run. Resets the FSM."""
        now_ms = int(time.monotonic() * 1000)

        self._fsm = PickFSM(self._config)
        self._fsm.start(now_ms)

        self._skill_name = skill_name
        self._skill_run_id = skill_run_id
        self._skill_start_ms = now_ms
        self._last_distance_m = None

        store = self._runtime.store
        await store.update_skill(
            self._arm_id,
            SkillInfo(name=skill_name, skill_run_id=skill_run_id, phase=PhaseId.APPROACH_PREGRASP),
        )
        await store.update_outcome(
            self._arm_id,
            OutcomeInfo(
                state=SkillOutcomeState.IN_PROGRESS,
                reason_code=None,
                needs_verify=not self._config.skip_verify_grasp,
            ),
        )
        await store.update_progress(
            self._arm_id,
            ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        )

        await self._publish(
            EventType.SKILL_STARTED,
            {
                "skill_run_id": skill_run_id,
                "skill_name": skill_name,
                "target_handle": target_handle,
            },
        )
        await self._publish(
            EventType.PHASE_ENTER,
            {"phase_id": int(PhaseId.APPROACH_PREGRASP)},
        )

    async def abort_skill(self) -> None:
        """Abort the current skill run (idempotent if not active)."""
        if not self._fsm.is_active:
            return

        now_ms = int(time.monotonic() * 1000)
        old_phase = self._fsm.phase
        self._fsm.abort(now_ms)

        await self._publish(EventType.PHASE_EXIT, {"phase_id": int(old_phase)})

        store = self._runtime.store
        await store.update_skill(
            self._arm_id,
            SkillInfo(
                name=self._skill_name,
                skill_run_id=self._skill_run_id,
                phase=PhaseId.DONE,
            ),
        )
        await store.update_outcome(
            self._arm_id,
            OutcomeInfo(
                state=SkillOutcomeState.FAILURE,
                reason_code=self._fsm.failure_code,
                needs_verify=False,
            ),
        )
        await self._publish(
            EventType.SKILL_FAILED,
            {
                "skill_run_id": self._skill_run_id,
                "reason": "abort",
                "failure_code": "UNSAFE_ABORT",
            },
        )

    async def tick(self) -> PhaseId | None:
        """
        One runner tick. Callable directly in tests.
        Returns current FSM phase, or None if not active.
        """
        if not self._fsm.is_active:
            return None

        # Use cached snapshot if available; fall back to building a new one
        snap = await self._runtime.store.get_latest_snapshot(self._arm_id)
        if snap is None:
            snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)

        now_ms = int(time.monotonic() * 1000)
        old_phase = self._fsm.advance(now_ms, snap.target, snap.perception, snap.act)

        if old_phase is not None:
            await self._publish(EventType.PHASE_EXIT, {"phase_id": int(old_phase)})
            store = self._runtime.store
            await store.update_skill(
                self._arm_id,
                SkillInfo(
                    name=self._skill_name,
                    skill_run_id=self._skill_run_id,
                    phase=self._fsm.phase,
                ),
            )
            await self._publish(EventType.PHASE_ENTER, {"phase_id": int(self._fsm.phase)})

            if self._fsm.phase == PhaseId.DONE:
                if self._fsm.outcome == SkillOutcomeState.SUCCESS:
                    await store.update_outcome(
                        self._arm_id,
                        OutcomeInfo(
                            state=SkillOutcomeState.SUCCESS,
                            reason_code=None,
                            needs_verify=False,
                        ),
                    )
                    await self._publish(
                        EventType.SKILL_SUCCEEDED,
                        {"skill_run_id": self._skill_run_id},
                    )
                else:
                    await store.update_outcome(
                        self._arm_id,
                        OutcomeInfo(
                            state=SkillOutcomeState.FAILURE,
                            reason_code=self._fsm.failure_code,
                            needs_verify=False,
                        ),
                    )
                    await self._publish(
                        EventType.SKILL_FAILED,
                        {
                            "skill_run_id": self._skill_run_id,
                            "failure_code": self._fsm.failure_code.value
                            if self._fsm.failure_code
                            else None,
                        },
                    )

        # Update progress every tick
        elapsed_ms = now_ms - self._skill_start_ms
        delta_distance = 0.0
        if snap.target is not None and self._last_distance_m is not None:
            delta_distance = snap.target.distance_m - self._last_distance_m
        if snap.target is not None:
            self._last_distance_m = snap.target.distance_m
        await self._runtime.store.update_progress(
            self._arm_id,
            ProgressInfo(elapsed_ms=elapsed_ms, no_progress_ms=0, delta_distance=delta_distance),
        )

        # Chunk scheduling
        if self._fsm.is_active and self._fsm.needs_chunk(snap.act):
            chunk = await self._chunk_fn(self._arm_id, self._fsm.phase, snap)
            if chunk is not None:
                await self._push_fn(chunk)

        return self._fsm.phase

    # --- Private ---

    async def _run_loop(self) -> None:
        period = 1.0 / self._config.runner_rate_hz
        while not self._stop_event.is_set():
            await self.tick()
            await asyncio.sleep(period)

    async def _publish(self, event_type: EventType, data: dict) -> None:
        event = EventEnvelope(
            event_id=self._runtime.bus.make_event_id(),
            type=event_type,
            ts_ms=int(time.monotonic() * 1000),
            arm_id=self._arm_id,
            data=data,
        )
        await self._runtime.bus.publish(event)
