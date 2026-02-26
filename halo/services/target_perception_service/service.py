from __future__ import annotations

import asyncio
import dataclasses
import re
import time
from typing import TYPE_CHECKING, Awaitable, Callable

from halo.contracts.enums import CommandType, PerceptionFailureCode, TrackingStatus
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import PerceptionInfo, TargetInfo
from halo.runtime.runtime import HALORuntime
from halo.services.target_perception_service.config import TargetPerceptionServiceConfig
from halo.services.target_perception_service.frame_buffer import CapturedFrame, FrameRingBuffer
from halo.services.target_perception_service.vlm_parser import VlmDetection, VlmScene

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

# observe_fn(arm_id, target_handle) -> TargetInfo | None
# Wraps the tracker + depth fusion steady-state path (10–30 Hz).
# Returns None when the target is momentarily lost.
ObserveFn = Callable[[str, str], Awaitable[TargetInfo | None]]

# vlm_fn(arm_id, image, known_handles) -> VlmScene
# Scene analysis — receives a camera frame and previously known handles.
# The service picks the matching target from the scene.
# Runs asynchronously, never on the fast-loop path.
VlmFn = Callable[[str, object, list[str]], Awaitable[VlmScene]]

# capture_fn(arm_id) -> CapturedFrame
# Grab the latest camera frame. Must be fast (sub-ms, from a pre-filled
# camera buffer) — it is called on every tick while VLM inference is active.
CaptureFn = Callable[[str], Awaitable[CapturedFrame]]

# tracker_update_fn(frame) -> TargetInfo | None
# Feed the next frame to an already-initialised tracker.  Returns updated
# TargetInfo, or None if the tracker lost the target on this frame.
TrackerUpdateFn = Callable[[CapturedFrame], Awaitable[TargetInfo | None]]

# tracker_factory_fn(frame, detection) -> (TargetInfo, TrackerUpdateFn)
# Initialise a tracker on *frame* using *detection* (bbox/centroid).
# Returns the initial TargetInfo and a bound update function that closes
# over the tracker state.
TrackerFactoryFn = Callable[[CapturedFrame, VlmDetection], Awaitable[tuple[TargetInfo, TrackerUpdateFn]]]


def _find_detection(target_handle: str, detections: list[VlmDetection]) -> VlmDetection | None:
    """Find a detection matching *target_handle*, with fuzzy fallback.

    1. Exact handle match.
    2. Fuzzy: strip trailing ``_NN`` suffix and match on prefix
       (e.g. ``black_cube_01`` matches ``black_cube_02``).
       If multiple candidates share the prefix, pick the first.
    """
    exact = next((d for d in detections if d.handle == target_handle), None)
    if exact is not None:
        return exact
    prefix = re.sub(r"_\d+$", "", target_handle)
    candidates = [d for d in detections if re.sub(r"_\d+$", "", d.handle) == prefix]
    return candidates[0] if candidates else None


class TargetPerceptionService:
    """
    Fast-loop perception service (default 10 Hz) with async VLM reacquisition
    and frame-buffer replay for seamless tracker switchover.

    Fast-loop pipeline (one tick):
        1. If no target handle: publish LOST, no hint.
        2. Capture a frame into the replay buffer (if VLM inference is active).
        3. Obtain an observation:
           a. If an *active tracker* is set (post-switchover): feed the
              captured frame to it.
           b. Else if *observe_fn* is provided: call it (self-fed tracker).
              Consume the VLM seed if observe returns None.
           c. Else: VLM-only mode — re-publish cached observation.
        4. Apply plausibility gates (obs_age, time_skew) → may invalidate hint.
        5. Update tracking status and failure code.
        6. Write TargetInfo + PerceptionInfo to store.
        7. Emit PERCEPTION_FAILURE / PERCEPTION_RECOVERED on state transitions.

    VLM + replay path (async, never on the fast-loop critical path):
        - Triggered by set_tracking_target(), request_refresh(), or hitting
          reacquire_fail_limit consecutive observe=None results.
        - At most one VLM task runs at a time; duplicate triggers are dropped.
        - While the VLM runs, every tick captures a frame into the replay
          buffer.  The existing tracker (observe_fn or active_tracker_fn)
          keeps publishing hints uninterrupted.
        - On VLM completion, the replay task initialises a new tracker on
          the first buffered frame using the VLM detection, then replays all
          subsequent frames (including any that arrived during replay).
        - Once caught up, the new tracker becomes the active tracker: tick()
          switches to feeding it fresh frames instead of calling observe_fn.
        - If capture_fn or tracker_factory_fn is not provided, the service
          falls back to the original synthetic-seed behaviour.

    Injected callables:
        async def observe_fn(arm_id, handle) -> TargetInfo | None
        async def vlm_fn(arm_id) -> VlmScene               # optional
        async def capture_fn(arm_id) -> CapturedFrame       # optional
        async def tracker_factory_fn(frame, det) -> (TargetInfo, update_fn)  # optional

    Lifecycle:
        svc = TargetPerceptionService(
            arm_id, runtime, observe_fn,
            vlm_fn=vlm_fn, capture_fn=capture_fn,
            tracker_factory_fn=tracker_factory_fn,
        )
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
        capture_fn: CaptureFn | None = None,
        tracker_factory_fn: TrackerFactoryFn | None = None,
        config: TargetPerceptionServiceConfig = TargetPerceptionServiceConfig(),
        run_logger: RunLogger | None = None,
    ) -> None:
        self._arm_id = arm_id
        self._runtime = runtime
        self._observe_fn = observe_fn
        self._vlm_fn = vlm_fn
        self._capture_fn = capture_fn
        self._tracker_factory_fn = tracker_factory_fn
        self._config = config
        self._run_logger = run_logger

        self._target_handle: str | None = None
        self._tracking_status = TrackingStatus.IDLE
        self._failure_code = PerceptionFailureCode.OK
        self._reacquire_fail_count = 0
        self._tracker_init_attempts = 0  # VLM+replay retry counter for tracker init
        self._vlm_job_pending = False
        self._awaiting_acquisition = False

        # VLM async state
        self._vlm_task: asyncio.Task | None = None
        self._vlm_seed: TargetInfo | None = None  # latest VLM result, consumed by tick()
        self._last_obs: TargetInfo | None = None  # VLM-only: last good observation to re-publish
        self._known_handles: list[str] = []  # handles from last VLM scene, for ID stability

        # Frame buffer + tracker switchover
        self._replay_buffer = FrameRingBuffer(max_size=config.frame_buffer_max_size)
        self._active_tracker_fn: TrackerUpdateFn | None = None

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
        self._replay_buffer.stop()
        self._replay_buffer.clear()
        self._active_tracker_fn = None
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
        self._tracker_init_attempts = 0
        self._failure_code = PerceptionFailureCode.OK
        self._vlm_seed = None
        self._last_obs = None
        self._awaiting_acquisition = True
        self._active_tracker_fn = None  # force back to observe_fn path
        self._cancel_vlm()
        self._spawn_vlm(emit_scene_described=False, for_new_target=True)

    async def clear_tracking_target(self) -> None:
        """Stop tracking. Cancels any running VLM job. Next tick publishes LOST."""
        self._cancel_vlm()
        self._vlm_seed = None
        self._last_obs = None
        self._target_handle = None
        self._tracking_status = TrackingStatus.LOST
        self._reacquire_fail_count = 0
        self._active_tracker_fn = None

    async def request_refresh(self, mode: str = "reacquire", reason: str = "") -> None:
        """
        Request a VLM scene analysis / reacquisition.

        *mode*:
        - ``"reacquire"`` (default): scene analysis **and** tracker
          reacquisition if a target is set.
        - ``"scene_only"``: scene analysis only — never attempts tracker
          init, even if a target is active (used by DESCRIBE_SCENE).
        """
        reacquire = mode != "scene_only" and self._target_handle is not None
        if reacquire:
            self._tracking_status = TrackingStatus.REACQUIRING
        self._vlm_job_pending = True
        self._spawn_vlm(for_new_target=reacquire)

    # --- Testable internal ---

    async def tick(self) -> None:
        """
        One perception tick. Callable directly in tests.
        See class docstring for the full pipeline.
        """
        if self._target_handle is None:
            await self._publish_state(
                target=None,
                tracking_status=self._tracking_status,
                failure_code=PerceptionFailureCode.OK,
            )
            return

        # --- Frame capture: push into replay buffer if VLM inference is active ---
        captured_frame: CapturedFrame | None = None
        if self._capture_fn is not None and self._replay_buffer.is_active:
            try:
                captured_frame = await self._capture_fn(self._arm_id)
                self._replay_buffer.push(captured_frame)
            except Exception:
                pass  # capture failure is non-critical

        # --- Obtain observation from the active tracking source ---

        if self._active_tracker_fn is not None:
            # Post-switchover: buffer-fed tracker.
            if captured_frame is None and self._capture_fn is not None:
                try:
                    captured_frame = await self._capture_fn(self._arm_id)
                except Exception:
                    captured_frame = None
            if captured_frame is not None:
                try:
                    obs = await self._active_tracker_fn(captured_frame)
                except Exception:
                    obs = None
            else:
                obs = None
            # Active tracker path: skip vlm_seed consumption (already used at init)
            return await self._apply_gates_and_publish(obs)

        if self._observe_fn is None:
            # VLM-only mode: no fast tracker.
            # Consume seed when it arrives; re-publish cached obs on every tick.
            if self._vlm_seed is not None:
                self._last_obs = dataclasses.replace(
                    self._vlm_seed,
                    obs_age_ms=0,
                    time_skew_ms=0,
                )
                self._vlm_seed = None
            if self._last_obs is not None:
                await self._update_success(
                    tracking_status=TrackingStatus.TRACKING,
                    target=self._last_obs,
                )
            elif self._tracking_status != TrackingStatus.LOST:
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

        await self._apply_gates_and_publish(obs)

    # --- Private: plausibility gates + publish ---

    async def _apply_gates_and_publish(self, obs: TargetInfo | None) -> None:
        """Apply plausibility gates to *obs* and publish the result."""
        if obs is None:
            # If tracker init retries are exhausted (LOST), don't overwrite.
            if self._tracking_status == TrackingStatus.LOST:
                return
            self._reacquire_fail_count += 1
            if self._reacquire_fail_count >= self._config.reacquire_fail_limit:
                self._spawn_vlm(for_new_target=True)
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
        Listen for COMMAND_ACCEPTED events and handle DESCRIBE_SCENE and
        TRACK_OBJECT commands.
        """
        while not self._stop_event.is_set():
            try:
                event: EventEnvelope = await asyncio.wait_for(self._cmd_queue.get(), timeout=0.05)
                if event.type == EventType.COMMAND_ACCEPTED:
                    cmd_type = event.data.get("command_type")
                    if cmd_type == CommandType.DESCRIBE_SCENE:
                        await self.request_refresh(mode="scene_only", reason="command:DESCRIBE_SCENE")
                    elif cmd_type == CommandType.TRACK_OBJECT:
                        handle = event.data.get("target_handle", "")
                        if handle:
                            await self.set_tracking_target(handle)
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
        await self._runtime.store.update_target_and_perception(
            self._arm_id,
            target,
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
        if self._awaiting_acquisition and tracking_status == TrackingStatus.TRACKING:
            self._awaiting_acquisition = False
            await self._emit_event(
                EventType.TARGET_ACQUIRED,
                {"target_handle": self._target_handle or ""},
            )

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

    def _spawn_vlm(self, *, emit_scene_described: bool = True, for_new_target: bool = False) -> None:
        """Spawn a VLM task if vlm_fn is configured and none is already running."""
        if self._vlm_fn is None:
            return
        if self._vlm_task is not None and not self._vlm_task.done():
            return  # already running — do not stack
        self._vlm_job_pending = True
        # Start buffering frames for replay if this VLM run is for a new target
        if for_new_target and self._capture_fn is not None:
            self._replay_buffer.start()
        target = self._target_handle if for_new_target else None
        self._vlm_task = asyncio.create_task(self._run_vlm(target, emit_scene_described=emit_scene_described))

    def _cancel_vlm(self) -> None:
        """Cancel the running VLM task (fire-and-forget; stop() awaits it properly)."""
        if self._vlm_task is not None and not self._vlm_task.done():
            self._vlm_task.cancel()
        self._vlm_task = None
        self._replay_buffer.stop()
        self._replay_buffer.clear()

    async def _run_vlm(
        self,
        target_handle: str | None,
        *,
        emit_scene_described: bool = True,
    ) -> None:
        """
        Background VLM coroutine. Runs asynchronously — never awaited by tick().

        Emits SCENE_DESCRIBED unless *emit_scene_described* is False (e.g. when
        the VLM run was triggered by set_tracking_target / TRACK_OBJECT — only
        TARGET_ACQUIRED matters in that case).
        If a target_handle is set, attempts to seed the tracker via
        frame-buffer replay.  Retries up to ``tracker_init_retries`` times
        (re-running VLM + replay each attempt).  Status stays REACQUIRING
        during retries; on final failure it transitions to LOST.
        """
        if self._vlm_fn is None:
            return
        try:
            # Capture a frame for the VLM to analyse.
            vlm_image: object = None
            if self._capture_fn is not None:
                try:
                    vlm_frame = await self._capture_fn(self._arm_id)
                    vlm_image = vlm_frame.image
                except Exception:
                    pass  # VLM will receive None — implementation decides how to handle

            t0 = time.monotonic()
            scene = await self._vlm_fn(self._arm_id, vlm_image, self._known_handles)
            inference_ms = int((time.monotonic() - t0) * 1000)

            self._known_handles = [d.handle for d in scene.detections]
            det_summary = [{"handle": d.handle, "label": d.label} for d in scene.detections]

            if emit_scene_described:
                await self._emit_event(
                    EventType.SCENE_DESCRIBED,
                    {
                        "target_handle": target_handle or "",
                        "scene": scene.scene,
                        "detections": det_summary,
                        "count": len(scene.detections),
                        "inference_ms": inference_ms,
                    },
                )
                if self._run_logger is not None:
                    self._run_logger.log_scene_described(
                        scene_text=scene.scene,
                        detections=det_summary,
                        image=vlm_image,
                        inference_ms=inference_ms,
                    )

            # If tracking a target, seed the tracker for reacquisition.
            # On failure, re-spawn the full VLM+buffer+replay pipeline
            # (up to tracker_init_retries total attempts). During retries
            # the status stays REACQUIRING; after exhaustion → LOST.
            if target_handle is not None:
                self._tracker_init_attempts += 1
                max_attempts = self._config.tracker_init_retries

                match = _find_detection(target_handle, scene.detections)
                if match is None:
                    if self._run_logger is not None:
                        got = [d.handle for d in scene.detections]
                        self._run_logger.log_tracker(
                            event="handle_not_found",
                            target_handle=target_handle,
                            detail=f"vlm returned {got} (attempt {self._tracker_init_attempts}/{max_attempts})",
                        )
                    self._replay_buffer.stop()
                    self._replay_buffer.clear()
                    await self._tracker_init_retry_or_lost(target_handle, max_attempts)
                    return

                replay_result = await self._replay_and_init_tracker(match)
                if replay_result is not None:
                    latest, update_fn = replay_result
                    self._active_tracker_fn = update_fn
                    self._vlm_seed = latest
                    self._reacquire_fail_count = 0
                    attempt_num = self._tracker_init_attempts
                    self._tracker_init_attempts = 0
                    if self._run_logger is not None:
                        self._run_logger.log_tracker(
                            event="init_ok",
                            target_handle=match.handle,
                            detail=f"center_px={latest.center_px} bbox={match.bbox}"
                            + (f" (attempt {attempt_num})" if attempt_num > 1 else ""),
                        )
                else:
                    if self._run_logger is not None:
                        self._run_logger.log_tracker(
                            event="init_failed",
                            target_handle=match.handle,
                            detail=f"replay returned None, bbox={match.bbox}"
                            f" (attempt {self._tracker_init_attempts}/{max_attempts})",
                        )
                    await self._tracker_init_retry_or_lost(target_handle, max_attempts)
        except asyncio.CancelledError:
            self._replay_buffer.stop()
            self._replay_buffer.clear()
            raise
        finally:
            # Don't clear vlm_job_pending if a retry was spawned (new _vlm_task
            # was created by _tracker_init_retry_or_lost).
            if self._vlm_task is None or self._vlm_task.done():
                self._vlm_job_pending = False

    async def _tracker_init_retry_or_lost(self, target_handle: str, max_attempts: int) -> None:
        """Re-spawn VLM for another attempt, or declare LOST if exhausted.

        Called from ``_run_vlm`` (a background task).  During retries the
        status is set to REACQUIRING without emitting an event.  On
        exhaustion the status is set to LOST **first**, then
        ``PERCEPTION_FAILURE`` is emitted so the planner snapshot already
        reflects the terminal state when it wakes.
        """
        if self._tracker_init_attempts < max_attempts:
            if self._run_logger is not None:
                self._run_logger.log_tracker(
                    event="retry",
                    target_handle=target_handle,
                    detail=f"scheduling attempt {self._tracker_init_attempts + 1}/{max_attempts}",
                )
            await self._publish_state(
                tracking_status=TrackingStatus.REACQUIRING,
                failure_code=PerceptionFailureCode.REACQUIRE_FAILED,
                target=None,
            )
            # Mark current task as done so _spawn_vlm can create a new one.
            self._vlm_task = None
            self._spawn_vlm(emit_scene_described=False, for_new_target=True)
        else:
            if self._run_logger is not None:
                self._run_logger.log_tracker(
                    event="init_exhausted",
                    target_handle=target_handle,
                    detail=f"all {max_attempts} attempts failed",
                )
            self._tracker_init_attempts = 0
            self._vlm_task = None  # allow finally to clear _vlm_job_pending
            self._vlm_job_pending = False  # no retry spawned — clear before publishing
            # Set LOST in the store first, then emit the event so the
            # planner snapshot already shows LOST when it wakes.
            await self._publish_state(
                tracking_status=TrackingStatus.LOST,
                failure_code=PerceptionFailureCode.REACQUIRE_FAILED,
                target=None,
            )
            await self._emit_event(
                EventType.PERCEPTION_FAILURE,
                {"failure_code": PerceptionFailureCode.REACQUIRE_FAILED.value},
            )

    async def _replay_and_init_tracker(self, detection: VlmDetection) -> tuple[TargetInfo, TrackerUpdateFn] | None:
        """Replay buffered frames through a new tracker initialised with *detection*.

        Returns ``(final_target_info, update_fn)`` on success, or ``None`` if
        the replay path is unavailable or fails.

        Runs inside ``_run_vlm()`` (a background task) so it does **not** block
        ``tick()``.  New frames keep arriving in the buffer via ``tick()`` during
        replay; we consume them incrementally until caught up.
        """
        if self._tracker_factory_fn is None or self._capture_fn is None:
            if self._run_logger is not None:
                missing = []
                if self._tracker_factory_fn is None:
                    missing.append("tracker_factory_fn")
                if self._capture_fn is None:
                    missing.append("capture_fn")
                self._run_logger.log_tracker(
                    event="replay_skip",
                    target_handle=detection.handle,
                    detail=f"missing {', '.join(missing)}",
                )
            self._replay_buffer.stop()
            self._replay_buffer.clear()
            return None

        read_idx = 0

        # Wait for at least one frame in the buffer.
        while True:
            frames, read_idx = self._replay_buffer.read_from(read_idx)
            if frames:
                break
            if not self._replay_buffer.is_active:
                # Buffer closed with no frames — capture one live frame.
                try:
                    live = await self._capture_fn(self._arm_id)
                    frames = [live]
                    break
                except Exception as exc:
                    if self._run_logger is not None:
                        self._run_logger.log_tracker(
                            event="replay_capture_failed",
                            target_handle=detection.handle,
                            detail=str(exc),
                        )
                    return None
            await asyncio.sleep(0)  # yield for tick() to push

        if self._run_logger is not None:
            self._run_logger.log_tracker(
                event="replay_start",
                target_handle=detection.handle,
                detail=f"frames={len(frames)} bbox={detection.bbox}",
            )

        # Initialise tracker on the first frame.
        try:
            latest, update_fn = await self._tracker_factory_fn(frames[0], detection)
        except Exception as exc:
            if self._run_logger is not None:
                self._run_logger.log_tracker(
                    event="replay_error",
                    target_handle=detection.handle,
                    detail=f"tracker_factory raised: {exc}",
                )
            self._replay_buffer.stop()
            self._replay_buffer.clear()
            return None

        # Replay remaining frames from the initial batch.
        for frame in frames[1:]:
            try:
                result = await update_fn(frame)
                if result is not None:
                    latest = result
            except Exception:
                break

        # Consume frames that arrived during replay (and keep going until caught up).
        while True:
            frames, read_idx = self._replay_buffer.read_from(read_idx)
            if not frames:
                # Yield once to let tick() push any pending frame.
                await asyncio.sleep(0)
                frames, read_idx = self._replay_buffer.read_from(read_idx)
                if not frames:
                    break  # truly caught up
            for frame in frames:
                try:
                    result = await update_fn(frame)
                    if result is not None:
                        latest = result
                except Exception:
                    break

        self._replay_buffer.stop()
        self._replay_buffer.clear()
        return latest, update_fn
