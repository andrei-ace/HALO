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
from halo.services.skill_runner_service.definitions import SkillRegistry, build_default_registry
from halo.services.skill_runner_service.engine import FsmEngine
from halo.services.skill_runner_service.queue import SkillQueue
from halo.services.skill_runner_service.skill_run import QueuedSkill, SkillRun
from halo.services.skill_runner_service.view_model import FsmViewModel, build_fsm_view_model

logger = logging.getLogger(__name__)

# Injected callables — decoupled from ACT model and ControlService instance
ChunkFn = Callable[
    [str, PhaseId, object],  # (arm_id, phase, PlannerSnapshot)
    Awaitable[ActionChunk | None],
]
PushFn = Callable[[ActionChunk], Awaitable[None]]


# --- Sim mode types ---

StartPickFn = Callable[[str, str], Awaitable[dict]]  # (arm_id, target_body) → server response
StartPlaceFn = Callable[[str, str, str], Awaitable[dict]]  # (arm_id, target_body, held_body) → server response
AbortPickFn = Callable[[], Awaitable[dict]]  # () → server response
SimPhaseFn = Callable[[], tuple[int, bool, str | None]]  # () → (phase_id, done, error)


class SkillRunnerService:
    """
    10-20 Hz asyncio loop owning the skill FSM engine, ACT chunk scheduling,
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
        start_place_fn: StartPlaceFn | None = None,
        abort_pick_fn: AbortPickFn | None = None,
        sim_phase_fn: SimPhaseFn | None = None,
        registry: SkillRegistry | None = None,
    ) -> None:
        act_mode = chunk_fn is not None or push_fn is not None
        sim_mode = start_pick_fn is not None or start_place_fn is not None

        if act_mode and sim_mode:
            raise ValueError("Cannot provide both ACT and sim callables")
        if not act_mode and not sim_mode:
            raise ValueError("Must provide either ACT (chunk_fn + push_fn) or sim callables")

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
        self._start_place_fn = start_place_fn
        self._abort_pick_fn = abort_pick_fn
        self._sim_phase_fn = sim_phase_fn

        # Engine-based FSM
        self._registry = registry or build_default_registry()
        self._queue = SkillQueue(max_size=config.max_queue_size)
        self._engine: FsmEngine | None = None
        self._active_run: SkillRun | None = None
        self._skill_name: SkillName | None = None
        self._skill_run_id: str | None = None
        self._active_target_handle: str | None = None
        self._skill_start_ms: int = 0
        self._last_distance_m: float | None = None

        self._sim_triggered: bool = False  # sim mode: True after start_pick/place_fn called
        self._sim_seen_active: bool = False  # True after telemetry shows done=False post-trigger
        self._sim_trigger_ms: int = 0  # monotonic ms when sim trajectory was triggered

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
        *,
        variant: str = "default",
        options: dict | None = None,
    ) -> None:
        """Begin a new skill run. If a skill is active, enqueue."""
        # Check registry for the skill
        defn = self._registry.get(skill_name, variant)
        if defn is None:
            await self._publish(
                EventType.SKILL_FAILED,
                {
                    "skill_run_id": skill_run_id,
                    "skill_name": skill_name,
                    "reason": "unsupported_skill",
                    "failure_code": SkillFailureCode.PLANNER_ABORT.value,
                },
            )
            # If a skill is already active, do not clobber it.
            if self._active_run is not None and self._active_run.is_active:
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
                    reason_code=SkillFailureCode.PLANNER_ABORT,
                    needs_verify=False,
                ),
            )
            self._active_target_handle = None
            return

        # If there's an active run, enqueue
        if self._active_run is not None and self._active_run.is_active:
            now_ms = int(time.monotonic() * 1000)
            queued = QueuedSkill(
                skill_name=skill_name,
                skill_run_id=skill_run_id,
                target_handle=target_handle,
                variant=variant,
                options=options or {},
                enqueued_at_ms=now_ms,
            )
            if not self._queue.enqueue(queued):
                await self._publish(
                    EventType.SKILL_FAILED,
                    {
                        "skill_run_id": skill_run_id,
                        "skill_name": skill_name,
                        "reason": "queue_full",
                        "failure_code": SkillFailureCode.PLANNER_ABORT.value,
                    },
                )
            return

        await self._activate_skill(skill_name, skill_run_id, target_handle, variant, defn, options=options)

    async def abort_skill(
        self, code: SkillFailureCode = SkillFailureCode.PLANNER_ABORT, *, clear_queue: bool = False
    ) -> None:
        """Abort the current skill run (idempotent if not active)."""
        if self._active_run is None or not self._active_run.is_active:
            return

        # Stop the sim trajectory first so the arm freezes immediately
        if self._sim_mode and self._abort_pick_fn is not None and self._sim_triggered:
            try:
                await self._abort_pick_fn()
            except Exception:
                logger.warning("abort_pick_fn failed", exc_info=True)

        now_ms = int(time.monotonic() * 1000)
        old_phase = self._active_run.phase_id
        self._engine.abort(self._active_run, now_ms, code)

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
                reason_code=self._active_run.failure_code,
                needs_verify=False,
            ),
        )
        await self._publish(
            EventType.SKILL_FAILED,
            {
                "skill_run_id": self._skill_run_id,
                "skill_name": self._skill_name,
                "reason": "abort",
                "failure_code": self._active_run.failure_code.value,
            },
        )
        self._active_target_handle = None

        if clear_queue:
            cleared = self._queue.clear()
            if cleared:
                logger.info("abort_skill: cleared %d queued skill(s)", cleared)
        else:
            # Auto-activate next from queue
            await self._activate_next_from_queue()

    async def tick(self) -> PhaseId | None:
        """One runner tick. Dispatches to ACT, sim, or track mode."""
        if self._active_run is not None and self._active_run.skill_name == SkillName.TRACK:
            return await self._tick_track()
        if self._sim_mode:
            return await self._tick_sim()
        return await self._tick_act()

    def get_view_model(self) -> FsmViewModel | None:
        now_ms = int(time.monotonic() * 1000)
        return build_fsm_view_model(self._active_run, self._queue, now_ms)

    # --- Backward-compatible properties for tests ---

    @property
    def _fsm(self):
        """Backward-compatible access to FSM state via active_run.

        Returns a lightweight proxy so existing tests that read _fsm.phase,
        _fsm.outcome, _fsm.failure_code, _fsm.is_active, etc. still work.
        """
        return _FsmProxy(self._active_run, self._engine, self._config)

    # --- Track mode tick ---

    async def _tick_track(self) -> PhaseId | None:
        if self._active_run is None or not self._active_run.is_active:
            return None

        snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)
        now_ms = int(time.monotonic() * 1000)
        run = self._active_run
        old_phase = self._engine.advance(
            run,
            now_ms,
            snap.target,
            snap.perception,
            snap.act,
            held_object_handle=snap.held_object_handle,
        )

        if old_phase is not None:
            await self._handle_transition(old_phase)
            if not run.is_active:
                return run.phase_id

        elapsed_ms = now_ms - self._skill_start_ms
        await self._runtime.store.update_progress(
            self._arm_id,
            ProgressInfo(elapsed_ms=elapsed_ms, no_progress_ms=0, delta_distance=0.0),
        )

        return self._active_run.phase_id

    # --- Sim mode tick (unified for PICK and PLACE) ---

    async def _tick_sim(self) -> PhaseId | None:
        if self._active_run is None or not self._active_run.is_active:
            return None

        now_ms = int(time.monotonic() * 1000)
        run = self._active_run
        is_place = self._skill_name == SkillName.PLACE

        # Phase 1: handler-driven gate (SELECT_GRASP / SELECT_PLACE)
        if not self._sim_triggered:
            snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)

            old_phase = self._engine.advance(
                run,
                now_ms,
                snap.target,
                snap.perception,
                snap.act,
                held_object_handle=snap.held_object_handle,
            )
            if old_phase is not None:
                await self._handle_transition(old_phase)
                if not run.is_active:
                    return run.phase_id

            # Gate passed — trigger sim trajectory
            if is_place:
                if run.phase_id == PhaseId.TRANSIT_PREPLACE:
                    logger.info("PLACE gate passed (TRANSIT_PREPLACE) — triggering sim trajectory")
                    await self._trigger_sim_trajectory(now_ms)
            else:
                if run.phase_id.value >= PhaseId.PLAN_APPROACH.value and run.phase_id != PhaseId.DONE:
                    await self._trigger_sim_trajectory(now_ms)
        else:
            # Phase 2: sim trajectory running — sync FSM from telemetry
            if self._sim_phase_fn is not None:
                phase_id, done, sim_error = self._sim_phase_fn()
            else:
                return run.phase_id

            # Stale telemetry guard: after triggering a new sim command,
            # the server resets done=False. Until we see that first
            # done=False frame, any done=True is leftover from the
            # previous skill's completed trajectory.
            if not done:
                if not self._sim_seen_active:
                    logger.info("Sim telemetry: first done=False frame received (phase_id=%d)", phase_id)
                self._sim_seen_active = True
            elif not self._sim_seen_active:
                stale_wait_ms = now_ms - self._sim_trigger_ms
                if stale_wait_ms < self._config.sim_stale_guard_timeout_ms:
                    return run.phase_id  # plausible stale — keep waiting
                logger.warning(
                    "Stale guard timeout (%d ms) — accepting done=True (phase_id=%d)",
                    stale_wait_ms,
                    phase_id,
                )
                self._sim_seen_active = True  # break out, fall through to failure detection

            # Sim server reported an error (e.g. NO_GRASP from failed VERIFY_GRASP).
            # The error persists across the return trajectory so we catch it here.
            if done and sim_error:
                fail_code = SkillFailureCode.PLACE_MISS if is_place else SkillFailureCode.NO_GRASP
                logger.warning("Sim error=%s (phase_id=%d) — treating as %s", sim_error, phase_id, fail_code.name)
                old_phase = run.phase_id
                self._engine.fail(run, now_ms, fail_code)
                await self._handle_transition(old_phase)
                return run.phase_id

            # Early failure detection (server planning failed or trajectory
            # ended before reaching the success phase).
            min_success_phase = PhaseId.RETREAT.value if is_place else PhaseId.VERIFY_GRASP.value
            if done and phase_id < min_success_phase:
                fail_code = SkillFailureCode.PLACE_MISS if is_place else SkillFailureCode.NO_GRASP
                logger.warning("Sim done=True with early phase_id=%d — treating as %s", phase_id, fail_code.name)
                old_phase = run.phase_id
                self._engine.fail(run, now_ms, fail_code)
                await self._handle_transition(old_phase)
                return run.phase_id

            sim_phase = PhaseId(phase_id) if not done else PhaseId.DONE
            old_phase = self._engine.sync_phase(run, now_ms, sim_phase)

            if old_phase is not None:
                await self._handle_transition(old_phase)
                if not run.is_active:
                    return run.phase_id

        # Update progress
        elapsed_ms = now_ms - self._skill_start_ms
        await self._runtime.store.update_progress(
            self._arm_id,
            ProgressInfo(elapsed_ms=elapsed_ms, no_progress_ms=0, delta_distance=0.0),
        )

        # Update ActInfo
        wrist = self._active_run.phase_id in WRIST_ACTIVE_PHASES
        await self._runtime.store.update_act(
            self._arm_id,
            ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=0, buffer_low=False, wrist_enabled=wrist),
        )

        return self._active_run.phase_id

    async def _trigger_sim_trajectory(self, now_ms: int) -> None:
        """Send the sim command (start_pick or start_place) to the server."""
        if self._active_target_handle is None:
            return

        if self._skill_name == SkillName.PLACE:
            if self._start_place_fn is None:
                logger.warning("PLACE in sim mode but start_place_fn not provided")
                return
            self._sim_triggered = True
            self._sim_seen_active = False
            self._sim_trigger_ms = now_ms
            await self._do_trigger_place(now_ms)
        else:
            if self._start_pick_fn is None:
                logger.warning("PICK in sim mode but start_pick_fn not provided")
                return
            self._sim_triggered = True
            self._sim_seen_active = False
            self._sim_trigger_ms = now_ms
            await self._do_trigger_pick(now_ms)

    async def _do_trigger_pick(self, now_ms: int) -> None:
        try:
            resp = await self._start_pick_fn(self._arm_id, self._active_target_handle)
            if resp.get("type") == "start_pick_error":
                logger.warning("start_pick_fn returned error: %s", resp.get("message"))
                old_phase = self._active_run.phase_id
                self._engine.fail(self._active_run, now_ms, SkillFailureCode.NO_GRASP)
                await self._handle_transition(old_phase)
        except BridgeTransportError:
            logger.warning("start_pick_fn bridge transport failed, aborting skill")
            old_phase = self._active_run.phase_id
            self._engine.abort(self._active_run, now_ms)
            await self._handle_transition(old_phase)

    async def _do_trigger_place(self, now_ms: int) -> None:
        # Determine which object is held
        snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)
        held_handle = snap.held_object_handle
        logger.info("start_place: target=%s held=%s", self._active_target_handle, held_handle)
        if not held_handle:
            logger.warning("start_place_fn: no held_object_handle in runtime state")
            old_phase = self._active_run.phase_id
            self._engine.fail(self._active_run, now_ms, SkillFailureCode.PLACE_MISS)
            await self._handle_transition(old_phase)
            return
        try:
            resp = await self._start_place_fn(self._arm_id, self._active_target_handle, held_handle)
            logger.info("start_place_fn response: %s", resp)
            if resp.get("type") == "start_place_error":
                logger.warning("start_place_fn returned error: %s", resp.get("message"))
                old_phase = self._active_run.phase_id
                self._engine.fail(self._active_run, now_ms, SkillFailureCode.PLACE_MISS)
                await self._handle_transition(old_phase)
        except BridgeTransportError:
            logger.warning("start_place_fn bridge transport failed, aborting skill")
            old_phase = self._active_run.phase_id
            self._engine.abort(self._active_run, now_ms)
            await self._handle_transition(old_phase)

    # --- Shared transition handling ---

    async def _handle_transition(self, old_phase: PhaseId) -> None:
        await self._publish(EventType.PHASE_EXIT, {"phase_id": int(old_phase)})
        store = self._runtime.store
        await store.update_skill(
            self._arm_id,
            SkillInfo(
                name=self._skill_name,
                skill_run_id=self._skill_run_id,
                phase=self._active_run.phase_id,
            ),
        )
        await self._publish(EventType.PHASE_ENTER, {"phase_id": int(self._active_run.phase_id)})

        if self._active_run.phase_id == PhaseId.DONE:
            if self._active_run.outcome == SkillOutcomeState.SUCCESS:
                if self._skill_name == SkillName.PICK:
                    await store.update_held_object_handle(self._arm_id, self._active_target_handle)
                elif self._skill_name == SkillName.PLACE:
                    await store.update_held_object_handle(self._arm_id, None)
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
                    {"skill_run_id": self._skill_run_id, "skill_name": self._skill_name},
                )
            else:
                await store.update_outcome(
                    self._arm_id,
                    OutcomeInfo(
                        state=SkillOutcomeState.FAILURE,
                        reason_code=self._active_run.failure_code,
                        needs_verify=False,
                    ),
                )
                await self._publish(
                    EventType.SKILL_FAILED,
                    {
                        "skill_run_id": self._skill_run_id,
                        "skill_name": self._skill_name,
                        "failure_code": (
                            self._active_run.failure_code.value if self._active_run.failure_code else None
                        ),
                        "trigger": self._active_run.failure_trigger,
                        "failed_phase": self._active_run.failure_phase,
                        "target_handle": self._active_run.target_handle,
                    },
                )
                # Clear queue on failure — stale follow-up commands are invalid
                cleared = self._queue.clear()
                if cleared:
                    logger.info("skill failed: cleared %d queued skill(s)", cleared)
            self._active_target_handle = None

            if self._active_run and self._active_run.outcome == SkillOutcomeState.SUCCESS:
                # Auto-activate next from queue only on success
                await self._activate_next_from_queue()

    async def _tick_act(self) -> PhaseId | None:
        if self._active_run is None or not self._active_run.is_active:
            return None

        snap = await self._runtime.store.get_latest_snapshot(self._arm_id)
        if snap is None:
            snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)

        now_ms = int(time.monotonic() * 1000)
        run = self._active_run
        old_phase = self._engine.advance(
            run,
            now_ms,
            snap.target,
            snap.perception,
            snap.act,
            held_object_handle=snap.held_object_handle,
        )

        if old_phase is not None:
            await self._handle_transition(old_phase)
            # Terminal transition may auto-activate a queued skill — don't
            # continue with stale snapshot data from the previous run.
            if not run.is_active:
                return run.phase_id

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
        if (
            self._active_run.is_active
            and self._engine is not None
            and self._engine.needs_chunk(self._active_run, snap.act)
        ):
            chunk = await self._chunk_fn(self._arm_id, self._active_run.phase_id, snap)
            if chunk is not None:
                await self._push_fn(chunk)

        return self._active_run.phase_id

    # --- Skill activation ---

    async def _activate_skill(
        self,
        skill_name: SkillName,
        skill_run_id: str,
        target_handle: str,
        variant: str,
        defn=None,
        *,
        options: dict | None = None,
    ) -> None:
        if defn is None:
            defn = self._registry.get(skill_name, variant)
        if defn is None:
            return

        now_ms = int(time.monotonic() * 1000)

        # Build engine + run
        self._engine = FsmEngine(
            graph=defn.graph,
            handlers=defn.handler_factory(),
            config=self._config,
            global_guards=defn.global_guard_factory(),
        )
        self._active_run = self._engine.create_run(now_ms, skill_run_id, target_handle, variant)
        if options:
            self._active_run.state_bag.update(options)

        self._skill_name = skill_name
        self._skill_run_id = skill_run_id
        self._active_target_handle = target_handle
        self._skill_start_ms = now_ms
        self._last_distance_m = None
        self._sim_triggered = False
        self._sim_seen_active = False
        self._sim_trigger_ms = 0

        initial_phase = self._active_run.phase_id
        store = self._runtime.store
        await store.update_skill(
            self._arm_id,
            SkillInfo(name=skill_name, skill_run_id=skill_run_id, phase=initial_phase),
        )
        await store.update_outcome(
            self._arm_id,
            OutcomeInfo(
                state=SkillOutcomeState.IN_PROGRESS,
                reason_code=None,
                needs_verify=False
                if skill_name in (SkillName.TRACK, SkillName.PLACE)
                else not self._config.skip_verify_grasp,
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
            {"phase_id": int(initial_phase)},
        )

    async def _activate_next_from_queue(self) -> None:
        queued = self._queue.dequeue()
        if queued is None:
            return
        defn = self._registry.get(queued.skill_name, queued.variant)
        if defn is None:
            return
        await self._activate_skill(
            queued.skill_name,
            queued.skill_run_id,
            queued.target_handle,
            queued.variant,
            defn,
            options=queued.options or None,
        )

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


class _FsmProxy:
    """Backward-compatible proxy so existing tests reading _fsm.phase etc. still work."""

    def __init__(self, run: SkillRun | None, engine: FsmEngine | None, config: SkillRunnerConfig) -> None:
        self._run = run
        self._engine = engine
        self._config = config

    @property
    def phase(self) -> PhaseId:
        if self._run is None:
            return PhaseId.IDLE
        return self._run.phase_id

    @property
    def outcome(self) -> SkillOutcomeState:
        if self._run is None:
            return SkillOutcomeState.IN_PROGRESS
        return self._run.outcome

    @property
    def failure_code(self) -> SkillFailureCode | None:
        if self._run is None:
            return None
        return self._run.failure_code

    @property
    def phase_start_ms(self) -> int:
        if self._run is None:
            return 0
        return self._run.phase_start_ms

    @property
    def is_active(self) -> bool:
        if self._run is None:
            return False
        return self._run.is_active

    @property
    def wrist_camera_active(self) -> bool:
        if self._run is None:
            return False
        return self._run.wrist_camera_active
