from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
import re
import time
from typing import TYPE_CHECKING, Awaitable, Callable

from halo.contracts.enums import CommandType, PerceptionFailureCode, SkillName, TrackingStatus
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import PerceptionInfo, TargetInfo
from halo.runtime.runtime import HALORuntime
from halo.services.target_perception_service.config import TargetPerceptionServiceConfig
from halo.services.target_perception_service.frame_buffer import CapturedFrame, FrameRingBuffer
from halo.services.target_perception_service.handle_match import (
    dedupe_detection_handles,
    find_detection_by_handle,
)
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
# Implementations may optionally accept ``target_handle=...`` as a 4th arg;
# service introspection handles both forms.
VlmFn = Callable[[str, object, list[str]], Awaitable[VlmScene]]
VlmFnWithTargetHandle = Callable[[str, object, list[str], str | None], Awaitable[VlmScene]]

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


def _stabilize_scene_for_tracked_target(
    scene: VlmScene,
    tracked_handle: str | None,
    tracked_center_px: tuple[float, float] | None,
    *,
    max_center_dist_px: float = 0.15,
) -> VlmScene:
    """Stabilise handle for the actively tracked object on a scene refresh.

    Applied only for scene VLM calls that start while tracking is already in
    ``TRACKING`` state. The *tracked_center_px* snapshot is taken at VLM call
    start so matching reflects where the object was when inference began.
    """
    if tracked_handle is None or tracked_center_px is None or not scene.detections:
        return scene
    if any(d.handle == tracked_handle for d in scene.detections):
        return scene  # already stable

    prefix = re.sub(r"_\d+$", "", tracked_handle)
    candidates: list[tuple[float, int]] = []
    for idx, det in enumerate(scene.detections):
        det_prefix = re.sub(r"_\d+$", "", det.handle)
        if det_prefix != prefix:
            continue
        dist = math.hypot(det.centroid[0] - tracked_center_px[0], det.centroid[1] - tracked_center_px[1])
        if dist <= max_center_dist_px:
            candidates.append((dist, idx))
    if not candidates:
        return scene

    candidates.sort()
    _, chosen_idx = candidates[0]
    updated = list(scene.detections)
    updated[chosen_idx] = dataclasses.replace(updated[chosen_idx], handle=tracked_handle)
    return dataclasses.replace(scene, detections=updated)


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
        vlm_fn: VlmFn | VlmFnWithTargetHandle | None = None,
        capture_fn: CaptureFn | None = None,
        tracker_factory_fn: TrackerFactoryFn | None = None,
        config: TargetPerceptionServiceConfig | None = None,
        run_logger: RunLogger | None = None,
    ) -> None:
        if config is None:
            config = TargetPerceptionServiceConfig()

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

        # VLM async state — dual-slot: "scene" and "track" can run in parallel
        self._vlm_scene_task: asyncio.Task | None = None
        self._vlm_scene_run_seq = 0
        self._vlm_scene_active_run_id = 0

        self._vlm_track_task: asyncio.Task | None = None
        self._vlm_track_run_seq = 0
        self._vlm_track_active_run_id = 0
        self._vlm_accepts_target_handle: bool | None = None
        self._vlm_seed: TargetInfo | None = None  # latest VLM result, consumed by tick()
        self._last_obs: TargetInfo | None = None  # VLM-only: last good observation to re-publish
        self._known_handles: list[str] = []  # handles from last VLM scene, for ID stability
        self._last_tracked_center_px: tuple[float, float] | None = None

        # Frame buffer + tracker switchover
        self._replay_buffer = FrameRingBuffer(max_size=config.frame_buffer_max_size)
        self._active_tracker_fn: TrackerUpdateFn | None = None
        self._pending_tracker_fn: TrackerUpdateFn | None = None  # new tracker catching up
        self._tick_active_consumed: int = 0
        self._tick_active_total: int = 0
        self._tick_pending_consumed: int = 0
        self._tick_pending_total: int = 0

        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._cmd_task: asyncio.Task | None = None
        self._cmd_queue: asyncio.Queue | None = None

    # --- Public API ---

    async def start(self) -> None:
        """Spawn the fast perception loop and command listener."""
        if self._loop_task is not None and not self._loop_task.done():
            return
        self._stop_event.clear()
        if self._cmd_queue is None:
            self._cmd_queue = self._runtime.bus.subscribe(self._arm_id)
        if self._cmd_task is None or self._cmd_task.done():
            self._cmd_task = asyncio.create_task(self._drain_commands())
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop, cancel any running VLM job, await shutdown."""
        self._stop_event.set()
        self._replay_buffer.stop()
        self._replay_buffer.clear()
        self._active_tracker_fn = None
        self._pending_tracker_fn = None
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
        await self._cancel_vlm()  # cancels both slots

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
        self._pending_tracker_fn = None
        self._last_tracked_center_px = None
        await self._cancel_vlm(slot="track")
        self._spawn_vlm(emit_scene_described=False, for_new_target=True, slot="track")

    async def clear_tracking_target(self) -> None:
        """Stop tracking. Cancels any running VLM job. Next tick publishes LOST."""
        await self._cancel_vlm()
        self._vlm_seed = None
        self._last_obs = None
        self._target_handle = None
        self._tracking_status = TrackingStatus.LOST
        self._reacquire_fail_count = 0
        self._awaiting_acquisition = False
        self._active_tracker_fn = None
        self._pending_tracker_fn = None
        self._last_tracked_center_px = None

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
        if self._vlm_fn is None:
            self._vlm_job_pending = False
            return
        self._vlm_job_pending = True
        if mode == "scene_only":
            self._spawn_vlm(for_new_target=False, slot="scene")
        else:
            self._spawn_vlm(for_new_target=reacquire, slot="track")

    # --- Testable internal ---

    async def _capture_batch(self) -> list[CapturedFrame]:
        """Drain the right number of frames from ``capture_fn`` per tick.

        Computes frames-per-tick from ``capture_source_fps / fast_loop_hz``
        (e.g. 30 / 10 = 3), capped at ``max_frames_per_tick``.

        This is intentionally **not** wall-clock based: tick processing time
        varies (pending tracker draining can take hundreds of ms), and
        compensating for that delay by reading extra frames causes the
        tracker to overshoot real-time.
        """
        if self._capture_fn is None:
            return []

        n_frames = max(
            1,
            min(
                round(self._config.capture_source_fps / self._config.fast_loop_hz),
                self._config.max_frames_per_tick,
            ),
        )

        frames: list[CapturedFrame] = []
        for _ in range(n_frames):
            try:
                frames.append(await self._capture_fn(self._arm_id))
            except Exception:
                break
        return frames

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

        # --- Capture fresh frames ---
        fresh_frames: list[CapturedFrame] = []
        has_tracker = self._active_tracker_fn is not None or self._pending_tracker_fn is not None
        need_capture = self._capture_fn is not None and (self._replay_buffer.is_active or has_tracker)
        if need_capture:
            fresh_frames = await self._capture_batch()

        # --- Push fresh frames into replay buffer (during VLM or pending catchup) ---
        if self._replay_buffer.is_active:
            for frame in fresh_frames:
                self._replay_buffer.push(frame)

        # --- Dual-tracker path ---
        if self._active_tracker_fn is not None or self._pending_tracker_fn is not None:
            obs = None
            self._tick_active_consumed = 0
            self._tick_active_total = len(fresh_frames)
            self._tick_pending_consumed = 0
            self._tick_pending_total = 0

            # 1. Feed fresh frames to active (old) tracker for live observations
            if self._active_tracker_fn is not None:
                for frame in fresh_frames:
                    try:
                        result = await self._active_tracker_fn(frame)
                        if result is not None:
                            obs = result
                        self._tick_active_consumed += 1
                    except Exception:
                        pass

            # 2. Drain pending (new) tracker's buffer
            if self._pending_tracker_fn is not None:
                buf_remaining = self._replay_buffer.remaining
                self._tick_pending_total = buf_remaining

                if buf_remaining > 0:
                    n = min(buf_remaining, self._config.max_frames_per_tick)
                    batch, _ = self._replay_buffer.read_from(self._replay_buffer.cursor)
                    pending_frames = batch[:n]
                    self._replay_buffer.advance_cursor(self._replay_buffer.cursor + len(pending_frames))
                    self._tick_pending_consumed = len(pending_frames)

                    pending_obs = None
                    for frame in pending_frames:
                        try:
                            result = await self._pending_tracker_fn(frame)
                            if result is not None:
                                pending_obs = result
                        except Exception:
                            pass

                    # If no active tracker, pending provides obs
                    if self._active_tracker_fn is None and pending_obs is not None:
                        obs = pending_obs

                # Check if pending tracker has caught up
                if self._replay_buffer.remaining == 0:
                    self._active_tracker_fn = self._pending_tracker_fn
                    self._pending_tracker_fn = None
                    self._replay_buffer.stop()
                    self._replay_buffer.clear()

            status_override = TrackingStatus.REACQUIRING if self._pending_tracker_fn is not None else None
            return await self._apply_gates_and_publish(obs, status_override=status_override)

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

    async def _apply_gates_and_publish(
        self, obs: TargetInfo | None, *, status_override: TrackingStatus | None = None
    ) -> None:
        """Apply plausibility gates to *obs* and publish the result."""
        if obs is None:
            # If tracker init retries are exhausted (LOST), don't overwrite.
            if self._tracking_status == TrackingStatus.LOST:
                return
            self._reacquire_fail_count += 1
            if self._reacquire_fail_count >= self._config.reacquire_fail_limit:
                self._spawn_vlm(for_new_target=True, slot="track")
                await self._update_failure(
                    tracking_status=TrackingStatus.REACQUIRING,
                    failure_code=PerceptionFailureCode.REACQUIRE_FAILED,
                    target=None,
                )
            elif self._last_obs is not None:
                # Brief tracker loss — hold last-known-good position with
                # degraded confidence so the TUI bbox stays stable.
                held = dataclasses.replace(self._last_obs, confidence=0.5, hint_valid=True)
                await self._publish_state(
                    target=held,
                    tracking_status=TrackingStatus.RELOCALIZING,
                    failure_code=PerceptionFailureCode.OK,
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
        if status_override is not None:
            tracking_status = status_override

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
        Listen for COMMAND_ACCEPTED events (DESCRIBE_SCENE) and
        SKILL_STARTED events (TRACK skill) to initiate tracking.
        """
        while not self._stop_event.is_set():
            try:
                event: EventEnvelope = await asyncio.wait_for(self._cmd_queue.get(), timeout=0.05)
                if event.type == EventType.COMMAND_ACCEPTED:
                    cmd_type = event.data.get("command_type")
                    if cmd_type == CommandType.DESCRIBE_SCENE:
                        await self.request_refresh(mode="scene_only", reason="command:DESCRIBE_SCENE")
                elif event.type == EventType.SKILL_STARTED:
                    if event.data.get("skill_name") == SkillName.TRACK:
                        handle = event.data.get("target_handle", "")
                        if handle:
                            await self.set_tracking_target(handle)
                elif event.type == EventType.SKILL_FAILED:
                    # Only tear down tracking on TRACK failure.  PICK failures
                    # (NO_GRASP, NO_PROGRESS, etc.) should preserve the active
                    # tracker so the planner can retry PICK without re-running
                    # TRACK.
                    if event.data.get("skill_name") == SkillName.TRACK:
                        await self.clear_tracking_target()
                    else:
                        self._awaiting_acquisition = False
                elif event.type == EventType.SKILL_SUCCEEDED:
                    # Only reset on TRACK completion.  PICK needs the active
                    # tracker to keep running (it was established by TRACK).
                    if event.data.get("skill_name") == SkillName.TRACK:
                        # TRACK succeeded — acquisition is done but keep the
                        # target handle and tracker alive for the upcoming PICK.
                        # Just reset _awaiting_acquisition so no duplicate
                        # TARGET_ACQUIRED can fire.
                        self._awaiting_acquisition = False
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
        if target is not None and target.center_px is not None:
            self._last_tracked_center_px = target.center_px
        await self._runtime.store.update_target_and_perception(
            self._arm_id,
            target,
            PerceptionInfo(
                tracking_status=tracking_status,
                failure_code=failure_code,
                reacquire_fail_count=self._reacquire_fail_count,
                vlm_job_pending=self._vlm_job_pending,
                active_buf_consumed=self._tick_active_consumed,
                active_buf_total=self._tick_active_total,
                pending_buf_consumed=self._tick_pending_consumed,
                pending_buf_total=self._tick_pending_total,
                has_pending_tracker=self._pending_tracker_fn is not None,
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

    def _spawn_vlm(
        self,
        *,
        emit_scene_described: bool = True,
        for_new_target: bool = False,
        slot: str = "track",
    ) -> bool:
        """Spawn a VLM task if vlm_fn is configured and the slot is free."""
        if self._vlm_fn is None:
            return False
        # Check slot-specific task
        if slot == "scene":
            if self._vlm_scene_task is not None and not self._vlm_scene_task.done():
                return False
            self._vlm_scene_run_seq += 1
            run_id = self._vlm_scene_run_seq
            self._vlm_scene_active_run_id = run_id
        else:
            if self._vlm_track_task is not None and not self._vlm_track_task.done():
                return False
            self._vlm_track_run_seq += 1
            run_id = self._vlm_track_run_seq
            self._vlm_track_active_run_id = run_id
        self._vlm_job_pending = True
        # Start buffering frames for replay if this VLM run is for a new target
        if for_new_target and self._capture_fn is not None:
            self._replay_buffer.start()
        target = self._target_handle if for_new_target else None
        stabilize_handle: str | None = None
        stabilize_center_px: tuple[float, float] | None = None
        if (
            not for_new_target
            and self._tracking_status == TrackingStatus.TRACKING
            and self._target_handle is not None
            and self._last_tracked_center_px is not None
        ):
            stabilize_handle = self._target_handle
            stabilize_center_px = self._last_tracked_center_px
        task = asyncio.create_task(
            self._run_vlm(
                run_id,
                target,
                slot=slot,
                emit_scene_described=emit_scene_described,
                stabilize_handle=stabilize_handle,
                stabilize_center_px=stabilize_center_px,
            )
        )
        if slot == "scene":
            self._vlm_scene_task = task
        else:
            self._vlm_track_task = task
        return True

    async def _cancel_vlm(self, slot: str | None = None) -> None:
        """Cancel VLM task(s). Default: cancel both. Pass slot='scene' or 'track' for one."""
        tasks_to_cancel: list[asyncio.Task] = []
        if slot is None or slot == "scene":
            if self._vlm_scene_task is not None and not self._vlm_scene_task.done():
                tasks_to_cancel.append(self._vlm_scene_task)
            self._vlm_scene_task = None
            self._vlm_scene_active_run_id = 0
        if slot is None or slot == "track":
            if self._vlm_track_task is not None and not self._vlm_track_task.done():
                tasks_to_cancel.append(self._vlm_track_task)
            self._vlm_track_task = None
            self._vlm_track_active_run_id = 0
            self._pending_tracker_fn = None
            self._replay_buffer.stop()
            self._replay_buffer.clear()
        if not self._vlm_scene_task and not self._vlm_track_task:
            self._vlm_job_pending = False
        for task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _active_run_id_for_slot(self, slot: str) -> int:
        return self._vlm_scene_active_run_id if slot == "scene" else self._vlm_track_active_run_id

    def _clear_slot_state(self, slot: str) -> None:
        if slot == "scene":
            self._vlm_scene_task = None
            self._vlm_scene_active_run_id = 0
        else:
            self._vlm_track_task = None
            self._vlm_track_active_run_id = 0

    async def _run_vlm(
        self,
        run_id: int,
        target_handle: str | None,
        *,
        slot: str = "track",
        emit_scene_described: bool = True,
        stabilize_handle: str | None = None,
        stabilize_center_px: tuple[float, float] | None = None,
    ) -> None:
        """
        Background VLM coroutine. Runs asynchronously — never awaited by tick().

        Emits SCENE_DESCRIBED unless *emit_scene_described* is False (e.g. when
        the VLM run was triggered by set_tracking_target / TRACK skill — only
        TARGET_ACQUIRED matters in that case).
        If a target_handle is set, attempts to seed the tracker via
        frame-buffer replay.  Retries up to ``tracker_init_retries`` times
        (re-running VLM + replay each attempt).  Status stays REACQUIRING
        during retries; on final failure it transitions to LOST.
        """
        if self._vlm_fn is None:
            return
        retry_spawned = False
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
            scene = await self._call_vlm(vlm_image, target_handle)
            inference_ms = int((time.monotonic() - t0) * 1000)
            if run_id != self._active_run_id_for_slot(slot):
                return
            scene = _stabilize_scene_for_tracked_target(scene, stabilize_handle, stabilize_center_px)
            scene = dataclasses.replace(scene, detections=dedupe_detection_handles(scene.detections))

            self._known_handles = [d.handle for d in scene.detections]
            det_log = [{"handle": d.handle, "label": d.label, "bbox": d.bbox} for d in scene.detections]

            if emit_scene_described:
                await self._emit_event(
                    EventType.SCENE_DESCRIBED,
                    {
                        "target_handle": target_handle or "",
                        "scene": scene.scene,
                        "detections": det_log,
                        "count": len(scene.detections),
                        "inference_ms": inference_ms,
                        "vlm_image": vlm_image,
                    },
                )
                if self._run_logger is not None:
                    await asyncio.to_thread(
                        self._run_logger.log_scene_described,
                        scene_text=scene.scene,
                        detections=det_log,
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

                match = find_detection_by_handle(target_handle, scene.detections)
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
                    retry_spawned = await self._tracker_init_retry_or_lost(
                        target_handle, max_attempts, run_id=run_id, slot=slot
                    )
                    return
                if match.handle != target_handle:
                    if self._run_logger is not None:
                        self._run_logger.log_tracker(
                            event="handle_alias",
                            target_handle=target_handle,
                            detail=f"fuzzy matched {match.handle}; publishing as {target_handle}",
                        )
                    # Fuzzy match selects the physical instance, but we keep the
                    # requested handle as the canonical ID for planner/UI consistency.
                    match = dataclasses.replace(match, handle=target_handle)

                replay_result = await self._replay_and_init_tracker(match)
                if replay_result is not None:
                    latest, update_fn = replay_result
                    self._pending_tracker_fn = update_fn
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
                    retry_spawned = await self._tracker_init_retry_or_lost(
                        target_handle, max_attempts, run_id=run_id, slot=slot
                    )
        except asyncio.CancelledError:
            if run_id == self._active_run_id_for_slot(slot):
                self._replay_buffer.stop()
                self._replay_buffer.clear()
            raise
        finally:
            # Only the currently active run may update shared slot state.
            if run_id == self._active_run_id_for_slot(slot) and not retry_spawned:
                self._clear_slot_state(slot)
                # vlm_job_pending is false only when neither slot is running
                if not self._vlm_scene_task and not self._vlm_track_task:
                    self._vlm_job_pending = False

    def _supports_vlm_target_handle(self) -> bool:
        if self._vlm_fn is None:
            return False
        if self._vlm_accepts_target_handle is not None:
            return self._vlm_accepts_target_handle
        try:
            sig = inspect.signature(self._vlm_fn)
            params = sig.parameters.values()
            supports = "target_handle" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        except (TypeError, ValueError):
            supports = False
        self._vlm_accepts_target_handle = supports
        return supports

    async def _call_vlm(self, vlm_image: object, target_handle: str | None) -> VlmScene:
        if self._vlm_fn is None:
            return VlmScene(scene="", detections=[])
        if self._supports_vlm_target_handle():
            return await self._vlm_fn(
                self._arm_id,
                vlm_image,
                self._known_handles,
                target_handle=target_handle,
            )
        return await self._vlm_fn(self._arm_id, vlm_image, self._known_handles)

    async def _tracker_init_retry_or_lost(
        self, target_handle: str, max_attempts: int, *, run_id: int, slot: str = "track"
    ) -> bool:
        """Re-spawn VLM for another attempt, or declare LOST if exhausted.

        Called from ``_run_vlm`` (a background task).  During retries the
        status is set to REACQUIRING without emitting an event.  On
        exhaustion the status is set to LOST **first**, then
        ``PERCEPTION_FAILURE`` is emitted so the planner snapshot already
        reflects the terminal state when it wakes.
        """
        if run_id != self._active_run_id_for_slot(slot):
            return False
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
            self._clear_slot_state(slot)
            self._spawn_vlm(emit_scene_described=False, for_new_target=True, slot=slot)
            return True
        else:
            if self._run_logger is not None:
                self._run_logger.log_tracker(
                    event="init_exhausted",
                    target_handle=target_handle,
                    detail=f"all {max_attempts} attempts failed",
                )
            self._tracker_init_attempts = 0
            self._clear_slot_state(slot)
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
            return False

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
                detail=f"buffered={len(frames)} bbox={detection.bbox}",
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

        # Keep buffer active — tick() will push fresh frames (for the new
        # tracker to catch up) and drain up to max_frames_per_tick per tick.
        # Frame 0 was consumed by init.
        self._replay_buffer.advance_cursor(1)

        return latest, update_fn
