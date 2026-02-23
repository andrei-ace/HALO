from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Awaitable, Callable

from halo.contracts.enums import CommandType, PerceptionFailureCode, TrackingStatus
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import PerceptionInfo, TargetInfo
from halo.runtime.runtime import HALORuntime
from halo.services.target_perception_service.config import TargetPerceptionServiceConfig
from halo.services.target_perception_service.vlm_parser import VlmScene

# observe_fn(arm_id, target_handle) -> TargetInfo | None
# Wraps the tracker + depth fusion steady-state path (10–30 Hz).
# Returns None when the target is momentarily lost.
ObserveFn = Callable[[str, str], Awaitable[TargetInfo | None]]

# vlm_fn(arm_id) -> VlmScene
# Scene analysis — returns all detections. The service picks the matching
# target from the scene. Runs asynchronously, never on the fast-loop path.
VlmFn = Callable[[str], Awaitable[VlmScene]]


class TargetPerceptionService:
    """
    Fast-loop perception service (default 10 Hz) with async VLM reacquisition.

    Fast-loop pipeline (one tick):
        1. If no target handle: publish LOST, no hint.
        2. Call observe_fn(arm_id, target_handle) — tracker steady-state.
        3. If observe returns None and a VLM seed is waiting: use the seed
           (simulates tracker picking up after VLM re-init). Consume seed.
        4. Apply plausibility gates (obs_age, time_skew) → may invalidate hint.
        5. Update tracking status and failure code.
        6. Write TargetInfo + PerceptionInfo to store.
        7. Emit PERCEPTION_FAILURE / PERCEPTION_RECOVERED on state transitions.

    VLM path (async, never on the fast-loop critical path):
        - Triggered by set_tracking_target(), request_refresh(), or hitting
          reacquire_fail_limit consecutive observe=None results.
        - At most one VLM task runs at a time; duplicate triggers are dropped.
        - On completion the result is stored in _vlm_seed; tick() consumes it
          on the next pass where observe_fn returns None.
        - vlm_fn is optional — omitting it disables VLM reacquisition entirely.

    Injected callables:
        async def observe_fn(arm_id, handle) -> TargetInfo | None
        async def vlm_fn(arm_id) -> VlmScene   # optional

    Lifecycle:
        svc = TargetPerceptionService(arm_id, runtime, observe_fn, vlm_fn=vlm_fn)
        await svc.set_tracking_target("cube-1")
        await svc.start()
        ...
        await svc.stop()
    """

    def __init__(
        self,
        arm_id: str,
        runtime: HALORuntime,
        observe_fn: ObserveFn | None = None,
        vlm_fn: VlmFn | None = None,
        config: TargetPerceptionServiceConfig = TargetPerceptionServiceConfig(),
    ) -> None:
        self._arm_id = arm_id
        self._runtime = runtime
        self._observe_fn = observe_fn
        self._vlm_fn = vlm_fn
        self._config = config

        self._target_handle: str | None = None
        self._tracking_status = TrackingStatus.LOST
        self._failure_code = PerceptionFailureCode.OK
        self._reacquire_fail_count = 0
        self._vlm_job_pending = False

        # VLM async state
        self._vlm_task: asyncio.Task | None = None
        self._vlm_seed: TargetInfo | None = None  # latest VLM result, consumed by tick()
        self._last_obs: TargetInfo | None = None   # VLM-only: last good observation to re-publish

        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._cmd_task: asyncio.Task | None = None
        self._cmd_queue: asyncio.Queue | None = None

    # --- Public API ---

    async def start(self) -> None:
        """Spawn the fast perception loop and command listener."""
        self._stop_event.clear()
        self._cmd_queue = self._runtime.bus.subscribe(self._arm_id)
        self._cmd_task = asyncio.create_task(self._drain_commands())
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop, cancel any running VLM job, await shutdown."""
        self._stop_event.set()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None
        if self._cmd_task is not None:
            self._cmd_task.cancel()
            try:
                await self._cmd_task
            except asyncio.CancelledError:
                pass
            self._cmd_task = None
        if self._cmd_queue is not None:
            self._runtime.bus.unsubscribe(self._arm_id, self._cmd_queue)
            self._cmd_queue = None
        # Cancel VLM task and await its termination to avoid dangling tasks.
        if self._vlm_task is not None:
            if not self._vlm_task.done():
                self._vlm_task.cancel()
                try:
                    await self._vlm_task
                except asyncio.CancelledError:
                    pass
            self._vlm_task = None

    async def set_tracking_target(self, target_handle: str) -> None:
        """
        Set the active target. Resets failure state and triggers VLM for
        initial acquisition (if vlm_fn is provided).
        """
        self._target_handle = target_handle
        self._reacquire_fail_count = 0
        self._failure_code = PerceptionFailureCode.OK
        self._vlm_seed = None
        self._last_obs = None
        self._cancel_vlm()
        self._spawn_vlm()

    async def clear_tracking_target(self) -> None:
        """Stop tracking. Cancels any running VLM job. Next tick publishes LOST."""
        self._cancel_vlm()
        self._vlm_seed = None
        self._last_obs = None
        self._target_handle = None
        self._tracking_status = TrackingStatus.LOST
        self._reacquire_fail_count = 0

    async def request_refresh(self, mode: str = "reacquire", reason: str = "") -> None:
        """
        Request a VLM scene analysis / reacquisition.

        Works with or without an active tracking target:
        - No target: runs VLM for scene analysis only, emits VLM_RESULT.
        - With target: additionally seeds the tracker for reacquisition.
        """
        if self._target_handle is not None:
            self._tracking_status = TrackingStatus.REACQUIRING
        self._vlm_job_pending = True
        self._spawn_vlm()

    # --- Testable internal ---

    async def tick(self) -> None:
        """
        One perception tick. Callable directly in tests.
        See class docstring for the full pipeline.
        """
        if self._target_handle is None:
            await self._publish_state(
                target=None,
                tracking_status=TrackingStatus.LOST,
                failure_code=PerceptionFailureCode.OK,
            )
            return

        if self._observe_fn is None:
            # VLM-only mode: no fast tracker.
            # Consume seed when it arrives; re-publish cached obs on every tick.
            if self._vlm_seed is not None:
                self._last_obs = dataclasses.replace(
                    self._vlm_seed, obs_age_ms=0, time_skew_ms=0,
                )
                self._vlm_seed = None
            if self._last_obs is not None:
                await self._update_success(
                    tracking_status=TrackingStatus.TRACKING, target=self._last_obs,
                )
            else:
                # Initial VLM call in flight — wait quietly.
                await self._publish_state(
                    target=None,
                    tracking_status=TrackingStatus.REACQUIRING,
                    failure_code=PerceptionFailureCode.OK,
                )
            return

        obs = await self._observe_fn(self._arm_id, self._target_handle)

        # If tracker lost the target but a VLM seed just arrived, use it.
        # This simulates the tracker being re-initialised by the VLM result.
        if obs is None and self._vlm_seed is not None:
            obs = self._vlm_seed
            self._vlm_seed = None

        if obs is None:
            self._reacquire_fail_count += 1
            if self._reacquire_fail_count >= self._config.reacquire_fail_limit:
                self._spawn_vlm()
                await self._update_failure(
                    tracking_status=TrackingStatus.REACQUIRING,
                    failure_code=PerceptionFailureCode.REACQUIRE_FAILED,
                    target=None,
                )
            else:
                await self._publish_state(
                    target=None,
                    tracking_status=TrackingStatus.RELOCALIZING,
                    failure_code=PerceptionFailureCode.OK,
                )
            return

        # Got an observation — apply plausibility gates
        self._reacquire_fail_count = 0
        failure_code = PerceptionFailureCode.OK
        hint_valid = obs.hint_valid

        if obs.obs_age_ms > self._config.obs_age_limit_ms:
            hint_valid = False
            failure_code = PerceptionFailureCode.DEPTH_INVALID
        elif abs(obs.time_skew_ms) > self._config.time_skew_limit_ms:
            hint_valid = False
            failure_code = PerceptionFailureCode.CALIB_INVALID

        if not hint_valid and obs.hint_valid:
            obs = dataclasses.replace(obs, hint_valid=False)

        tracking_status = TrackingStatus.TRACKING if hint_valid else TrackingStatus.RELOCALIZING

        if tracking_status == TrackingStatus.TRACKING:
            self._vlm_job_pending = False

        if failure_code != PerceptionFailureCode.OK:
            await self._update_failure(
                tracking_status=tracking_status,
                failure_code=failure_code,
                target=obs,
            )
        else:
            await self._update_success(tracking_status=tracking_status, target=obs)

    # --- Private: command listener ---

    async def _drain_commands(self) -> None:
        """
        Listen for COMMAND_ACCEPTED events and trigger VLM reacquisition when
        a REQUEST_PERCEPTION_REFRESH command is accepted by the router.
        """
        while not self._stop_event.is_set():
            try:
                event: EventEnvelope = await asyncio.wait_for(
                    self._cmd_queue.get(), timeout=0.05
                )
                if (
                    event.type == EventType.COMMAND_ACCEPTED
                    and event.data.get("command_type") == CommandType.REQUEST_PERCEPTION_REFRESH
                ):
                    await self.request_refresh(reason="command:REQUEST_PERCEPTION_REFRESH")
            except asyncio.TimeoutError:
                continue

    # --- Private: fast loop ---

    async def _run_loop(self) -> None:
        period = 1.0 / self._config.fast_loop_hz
        while not self._stop_event.is_set():
            await self.tick()
            await asyncio.sleep(period)

    async def _publish_state(
        self,
        target: TargetInfo | None,
        tracking_status: TrackingStatus,
        failure_code: PerceptionFailureCode,
    ) -> None:
        self._tracking_status = tracking_status
        self._failure_code = failure_code
        await self._runtime.store.update_target(self._arm_id, target)
        await self._runtime.store.update_perception(
            self._arm_id,
            PerceptionInfo(
                tracking_status=tracking_status,
                failure_code=failure_code,
                reacquire_fail_count=self._reacquire_fail_count,
                vlm_job_pending=self._vlm_job_pending,
            ),
        )

    async def _update_success(
        self,
        tracking_status: TrackingStatus,
        target: TargetInfo,
    ) -> None:
        was_failing = self._failure_code != PerceptionFailureCode.OK
        await self._publish_state(
            target=target,
            tracking_status=tracking_status,
            failure_code=PerceptionFailureCode.OK,
        )
        if was_failing:
            await self._emit_event(EventType.PERCEPTION_RECOVERED, {})

    async def _update_failure(
        self,
        tracking_status: TrackingStatus,
        failure_code: PerceptionFailureCode,
        target: TargetInfo | None,
    ) -> None:
        was_ok = self._failure_code == PerceptionFailureCode.OK
        await self._publish_state(
            target=target,
            tracking_status=tracking_status,
            failure_code=failure_code,
        )
        if was_ok:
            await self._emit_event(
                EventType.PERCEPTION_FAILURE,
                {"failure_code": failure_code.value},
            )

    async def _emit_event(self, event_type: EventType, data: dict) -> None:
        event = EventEnvelope(
            event_id=self._runtime.bus.make_event_id(),
            type=event_type,
            ts_ms=int(time.time() * 1000),
            arm_id=self._arm_id,
            data=data,
        )
        await self._runtime.bus.publish(event)

    # --- Private: VLM async path ---

    def _spawn_vlm(self) -> None:
        """Spawn a VLM task if vlm_fn is configured and none is already running."""
        if self._vlm_fn is None:
            return
        if self._vlm_task is not None and not self._vlm_task.done():
            return  # already running — do not stack
        self._vlm_job_pending = True
        self._vlm_task = asyncio.create_task(self._run_vlm(self._target_handle))

    def _cancel_vlm(self) -> None:
        """Cancel the running VLM task (fire-and-forget; stop() awaits it properly)."""
        if self._vlm_task is not None and not self._vlm_task.done():
            self._vlm_task.cancel()
        self._vlm_task = None

    async def _run_vlm(self, target_handle: str | None) -> None:
        """
        Background VLM coroutine. Runs asynchronously — never awaited by tick().

        Always emits VLM_RESULT with full scene data.
        If a target_handle is set, also seeds the tracker for reacquisition.
        """
        if self._vlm_fn is None:
            return
        try:
            t0 = time.monotonic()
            scene = await self._vlm_fn(self._arm_id)
            inference_ms = int((time.monotonic() - t0) * 1000)

            det_summary = [
                {"handle": d.handle, "label": d.label, "bbox": list(d.bbox)}
                for d in scene.detections
            ]

            # Always emit scene analysis result.
            await self._emit_event(
                EventType.VLM_RESULT,
                {
                    "target_handle": target_handle or "",
                    "scene": scene.scene,
                    "detections": det_summary,
                    "count": len(scene.detections),
                    "inference_ms": inference_ms,
                },
            )

            # If tracking a target, seed the tracker for reacquisition.
            if target_handle is not None:
                match = next(
                    (d for d in scene.detections if d.handle == target_handle), None,
                )
                if match is None and scene.detections:
                    match = scene.detections[0]
                if match is not None:
                    self._vlm_seed = TargetInfo(
                        handle=target_handle,
                        hint_valid=True,
                        confidence=1.0,
                        obs_age_ms=0,
                        time_skew_ms=0,
                        delta_xyz_ee=(0.0, 0.0, 0.0),
                        distance_m=0.0,
                    )
                    self._reacquire_fail_count = 0
        except asyncio.CancelledError:
            raise
        finally:
            self._vlm_job_pending = False
