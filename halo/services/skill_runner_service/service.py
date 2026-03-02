from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from halo.bridge import BridgeTransportError
from halo.contracts.actions import ActionChunk, JointPositionAction, JointPositionChunk
from halo.contracts.enums import WRIST_ACTIVE_PHASES, ActStatus, PhaseId, SkillFailureCode, SkillName, SkillOutcomeState
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


# --- Teacher mode types ---


@dataclass(frozen=True)
class TeacherStepResult:
    """Result from a single teacher step (server-side)."""

    phase_id: int
    done: bool
    action: tuple[float, ...]  # 6D joint-position that was applied on server


TeacherStepFn = Callable[[str], Awaitable[TeacherStepResult]]  # (arm_id,) → result
JointPushFn = Callable[[JointPositionChunk], Awaitable[None]]


class SkillRunnerService:
    """
    10–20 Hz asyncio loop owning the Pick-skill FSM, ACT chunk scheduling,
    and PHASE_ENTER/EXIT/SKILL_* event publishing.

    Supports two modes:
        - ACT mode (default): chunk_fn + push_fn drive actions
        - Teacher mode: teacher_step_fn + joint_push_fn drive actions

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
        teacher_step_fn: TeacherStepFn | None = None,
        joint_push_fn: JointPushFn | None = None,
    ) -> None:
        act_mode = chunk_fn is not None or push_fn is not None
        teacher_mode = teacher_step_fn is not None

        if act_mode and teacher_mode:
            raise ValueError("Cannot provide both ACT (chunk_fn/push_fn) and teacher (teacher_step_fn) callables")
        if not act_mode and not teacher_mode:
            raise ValueError("Must provide either ACT (chunk_fn + push_fn) or teacher (teacher_step_fn) callables")

        if act_mode:
            if chunk_fn is None or push_fn is None:
                raise ValueError("ACT mode requires both chunk_fn and push_fn")

        self._arm_id = arm_id
        self._runtime = runtime
        self._config = config
        self._teacher_mode = teacher_mode

        # ACT mode callables
        self._chunk_fn = chunk_fn
        self._push_fn = push_fn

        # Teacher mode callables
        self._teacher_step_fn = teacher_step_fn
        self._joint_push_fn = joint_push_fn

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
        """One runner tick. Dispatches to ACT or teacher mode."""
        if self._teacher_mode:
            return await self._tick_teacher()
        return await self._tick_act()

    # --- Teacher mode tick ---

    async def _tick_teacher(self) -> PhaseId | None:
        """Teacher-mode tick: call teacher_step_fn, sync FSM phase, push joint chunk."""
        if not self._fsm.is_active:
            return None

        now_ms = int(time.monotonic() * 1000)

        try:
            result = await self._teacher_step_fn(self._arm_id)
        except BridgeTransportError:
            logger.warning("SkillRunner: teacher_step_fn bridge transport failed, aborting skill")
            old_phase = self._fsm.phase
            self._fsm.abort(now_ms)
            await self._handle_transition(old_phase)
            return self._fsm.phase

        # Sync FSM phase from teacher
        teacher_phase = PhaseId(result.phase_id) if not result.done else PhaseId.DONE
        old_phase = self._fsm.sync_phase(now_ms, teacher_phase)

        if old_phase is not None:
            await self._handle_transition(old_phase)

        # Push joint-position chunk
        if self._joint_push_fn is not None and self._fsm.is_active:
            chunk = JointPositionChunk(
                chunk_id=f"teacher-{now_ms}",
                arm_id=self._arm_id,
                phase_id=self._fsm.phase,
                actions=(JointPositionAction(values=result.action),),
                ts_ms=now_ms,
            )
            await self._joint_push_fn(chunk)

        # Update progress (no delta_distance in teacher mode)
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
