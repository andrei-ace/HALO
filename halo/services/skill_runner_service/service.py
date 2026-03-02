from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from halo.bridge import BridgeTransportError
from halo.contracts.actions import ActionChunk
from halo.contracts.enums import (
    WRIST_ACTIVE_PHASES,
    ActStatus,
    PhaseId,
    SkillFailureCode,
    SkillName,
    SkillOutcomeState,
)
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import ActInfo, OutcomeInfo, ProgressInfo, SkillInfo
from halo.runtime.runtime import HALORuntime
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.fsm import PickFSM

logger = logging.getLogger(__name__)

# Injected callables — decoupled from ACT model and ControlService instance
ChunkFn = Callable[
    [str, PhaseId, object],  # (arm_id, phase, PlannerSnapshot)
    Awaitable[ActionChunk | None],
]
PushFn = Callable[[ActionChunk], Awaitable[None]]


# --- Sim mode types ---

StartPickFn = Callable[[str, str], Awaitable[dict]]  # (arm_id, target_body) → server response
SimPhaseFn = Callable[[], tuple[int, bool]]  # () → (phase_id, done)


class SkillRunnerService:
    """
    10–20 Hz asyncio loop owning the Pick-skill FSM, ACT chunk scheduling,
    and PHASE_ENTER/EXIT/SKILL_* event publishing.

    Supports two modes:
        - ACT mode (default): chunk_fn + push_fn drive actions
        - Sim mode: start_pick_fn + sim_phase_fn drive actions (server-side trajectory)

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
        chunk_fn: ChunkFn | None = None,
        push_fn: PushFn | None = None,
        config: SkillRunnerConfig = SkillRunnerConfig(),
        *,
        start_pick_fn: StartPickFn | None = None,
        sim_phase_fn: SimPhaseFn | None = None,
    ) -> None:
        act_mode = chunk_fn is not None or push_fn is not None
        sim_mode = start_pick_fn is not None

        if act_mode and sim_mode:
            raise ValueError("Cannot provide both ACT (chunk_fn/push_fn) and sim (start_pick_fn) callables")
        if not act_mode and not sim_mode:
            raise ValueError("Must provide either ACT (chunk_fn + push_fn) or sim (start_pick_fn) callables")

        if act_mode:
            if chunk_fn is None or push_fn is None:
                raise ValueError("ACT mode requires both chunk_fn and push_fn")

        self._arm_id = arm_id
        self._runtime = runtime
        self._config = config
        self._sim_mode = sim_mode

        # ACT mode callables
        self._chunk_fn = chunk_fn
        self._push_fn = push_fn

        # Sim mode callables
        self._start_pick_fn = start_pick_fn
        self._sim_phase_fn = sim_phase_fn

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
        if skill_name != SkillName.PICK:
            await self._publish(
                EventType.SKILL_FAILED,
                {
                    "skill_run_id": skill_run_id,
                    "reason": "unsupported_skill",
                    "failure_code": SkillFailureCode.UNSAFE_ABORT.value,
                },
            )
            # If a skill is already active, do not clobber it.
            if self._fsm.is_active:
                return
            await self._runtime.store.update_skill(
                self._arm_id,
                SkillInfo(
                    name=skill_name,
                    skill_run_id=skill_run_id,
                    phase=PhaseId.DONE,
                ),
            )
            await self._runtime.store.update_outcome(
                self._arm_id,
                OutcomeInfo(
                    state=SkillOutcomeState.FAILURE,
                    reason_code=SkillFailureCode.UNSAFE_ABORT,
                    needs_verify=False,
                ),
            )
            return
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
            SkillInfo(name=skill_name, skill_run_id=skill_run_id, phase=PhaseId.SELECT_GRASP),
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
            {"phase_id": int(PhaseId.SELECT_GRASP)},
        )

        # In sim mode, trigger trajectory planning on the server
        if self._sim_mode and self._start_pick_fn is not None:
            try:
                resp = await self._start_pick_fn(self._arm_id, target_handle)
                if resp.get("type") == "start_pick_error":
                    logger.warning("start_pick_fn returned error: %s", resp.get("message"))
                    old_phase = self._fsm.phase
                    self._fsm.fail(now_ms, SkillFailureCode.NO_GRASP)
                    await self._handle_transition(old_phase)
            except BridgeTransportError:
                logger.warning("start_pick_fn bridge transport failed, aborting skill")
                old_phase = self._fsm.phase
                self._fsm.abort(now_ms)
                await self._handle_transition(old_phase)

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
        """One runner tick. Dispatches to ACT or sim mode."""
        if self._sim_mode:
            return await self._tick_sim()
        return await self._tick_act()

    # --- Sim mode tick ---

    async def _tick_sim(self) -> PhaseId | None:
        """Sim-mode tick: read phase from sim_phase_fn, sync FSM, update progress."""
        if not self._fsm.is_active:
            return None

        now_ms = int(time.monotonic() * 1000)

        # Read phase from sim server telemetry
        if self._sim_phase_fn is not None:
            phase_id, done = self._sim_phase_fn()
        else:
            return self._fsm.phase

        # Sync FSM phase from sim
        sim_phase = PhaseId(phase_id) if not done else PhaseId.DONE
        old_phase = self._fsm.sync_phase(now_ms, sim_phase)

        if old_phase is not None:
            await self._handle_transition(old_phase)

        # Update progress (no delta_distance in sim mode)
        elapsed_ms = now_ms - self._skill_start_ms
        await self._runtime.store.update_progress(
            self._arm_id,
            ProgressInfo(elapsed_ms=elapsed_ms, no_progress_ms=0, delta_distance=0.0),
        )

        # Update ActInfo
        wrist = self._fsm.phase in WRIST_ACTIVE_PHASES
        await self._runtime.store.update_act(
            self._arm_id,
            ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=0, buffer_low=False, wrist_enabled=wrist),
        )

        return self._fsm.phase

    # --- Shared transition handling ---

    async def _handle_transition(self, old_phase: PhaseId) -> None:
        """Handle FSM transition: publish events, update store, emit SKILL_SUCCEEDED/FAILED."""
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
                        "failure_code": self._fsm.failure_code.value if self._fsm.failure_code else None,
                    },
                )

    async def _tick_act(self) -> PhaseId | None:
        """ACT-mode tick: FSM advance + chunk scheduling."""
        if not self._fsm.is_active:
            return None

        snap = await self._runtime.store.get_latest_snapshot(self._arm_id)
        if snap is None:
            snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)

        now_ms = int(time.monotonic() * 1000)
        old_phase = self._fsm.advance(now_ms, snap.target, snap.perception, snap.act)

        if old_phase is not None:
            await self._handle_transition(old_phase)

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
            ts_ms=int(time.time() * 1000),
            arm_id=self._arm_id,
            data=data,
        )
        await self._runtime.bus.publish(event)
