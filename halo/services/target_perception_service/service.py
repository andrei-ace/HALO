from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Awaitable, Callable

from halo.contracts.enums import PerceptionFailureCode, TrackingStatus
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import PerceptionInfo, TargetInfo
from halo.runtime.runtime import HALORuntime
from halo.services.target_perception_service.config import TargetPerceptionServiceConfig

# observe_fn(arm_id, target_handle) -> TargetInfo | None
# Returns None when the target cannot be observed (lost, occluded, etc.).
# Mock implementations return fixed/computed data; real implementations
# call tracker + depth fusion + plausibility pre-filter.
ObserveFn = Callable[[str, str], Awaitable[TargetInfo | None]]


class TargetPerceptionService:
    """
    Fast-loop perception service (default 10 Hz). Maintains a track on the
    active target and publishes fused TargetInfo + PerceptionInfo to
    RuntimeStateStore.

    Pipeline (one tick):
        1. If no target handle: publish LOST, no hint.
        2. Call observe_fn(arm_id, target_handle).
        3. Apply plausibility gates (obs_age, time_skew) → may invalidate hint.
        4. Update tracking status and failure code.
        5. Write TargetInfo + PerceptionInfo to store.
        6. Emit PERCEPTION_FAILURE / PERCEPTION_RECOVERED on state transitions.

    observe_fn is the only coupling to the sensor/model stack:
        async def observe_fn(arm_id: str, handle: str) -> TargetInfo | None: ...

    For v0, inject a mock that returns fixed data. Real implementations wire in
    ZED capture + SAM tracker + depth fusion.

    Lifecycle:
        svc = TargetPerceptionService(arm_id, runtime, observe_fn)
        await svc.set_tracking_target("cube-1")
        await svc.start()
        ...
        await svc.stop()
    """

    def __init__(
        self,
        arm_id: str,
        runtime: HALORuntime,
        observe_fn: ObserveFn,
        config: TargetPerceptionServiceConfig = TargetPerceptionServiceConfig(),
    ) -> None:
        self._arm_id = arm_id
        self._runtime = runtime
        self._observe_fn = observe_fn
        self._config = config

        self._target_handle: str | None = None
        self._tracking_status = TrackingStatus.LOST
        self._failure_code = PerceptionFailureCode.OK
        self._reacquire_fail_count = 0
        self._vlm_job_pending = False

        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None

    # --- Public API ---

    async def start(self) -> None:
        """Spawn the fast perception loop."""
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop and await clean shutdown."""
        self._stop_event.set()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None

    async def set_tracking_target(self, target_handle: str) -> None:
        """Set the active target. Resets failure state and fail counter."""
        self._target_handle = target_handle
        self._reacquire_fail_count = 0
        self._failure_code = PerceptionFailureCode.OK
        self._vlm_job_pending = False

    async def clear_tracking_target(self) -> None:
        """Stop tracking. Next tick publishes LOST with no hint."""
        self._target_handle = None
        self._tracking_status = TrackingStatus.LOST
        self._reacquire_fail_count = 0

    async def request_refresh(self, mode: str = "reacquire", reason: str = "") -> None:
        """
        Request a forced reacquisition (e.g. planner suspects target has moved).
        Sets status to REACQUIRING and marks a VLM job pending.
        The next successful observe clears both flags.
        """
        self._tracking_status = TrackingStatus.REACQUIRING
        self._vlm_job_pending = True

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

        obs = await self._observe_fn(self._arm_id, self._target_handle)

        if obs is None:
            self._reacquire_fail_count += 1
            if self._reacquire_fail_count >= self._config.reacquire_fail_limit:
                await self._update_failure(
                    tracking_status=TrackingStatus.REACQUIRING,
                    failure_code=PerceptionFailureCode.REACQUIRE_FAILED,
                    target=None,
                )
            else:
                # Transient loss — relocalizing, not yet a hard failure
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

    # --- Private ---

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
            ts_ms=int(time.monotonic() * 1000),
            arm_id=self._arm_id,
            data=data,
        )
        await self._runtime.bus.publish(event)
