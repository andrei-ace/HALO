"""Tests for TargetPerceptionService: tick, lifecycle, plausibility gates, events."""

import asyncio
import dataclasses

import pytest

from halo.contracts.enums import CommandType, PerceptionFailureCode, TrackingStatus
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import TargetInfo
from halo.runtime.runtime import HALORuntime
from halo.services.target_perception_service.config import TargetPerceptionServiceConfig
from halo.services.target_perception_service.frame_buffer import CapturedFrame
from halo.services.target_perception_service.mock_fns import make_mock_capture_fn, make_mock_tracker_factory_fn
from halo.services.target_perception_service.service import (
    CaptureFn,
    ObserveFn,
    TargetPerceptionService,
    TrackerFactoryFn,
    VlmFn,
    _stabilize_scene_for_tracked_target,
)
from halo.services.target_perception_service.vlm_parser import VlmDetection, VlmScene

ARM = "arm0"


@pytest.fixture
def rt() -> HALORuntime:
    r = HALORuntime()
    r.register_arm(ARM)
    return r


def _cfg(**kwargs) -> TargetPerceptionServiceConfig:
    kwargs.setdefault("fast_loop_hz", 10.0)
    return TargetPerceptionServiceConfig(**kwargs)


def _good_hint(handle: str = "cube-1", distance_m: float = 0.3) -> TargetInfo:
    return TargetInfo(
        handle=handle,
        hint_valid=True,
        confidence=0.95,
        obs_age_ms=5,
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, distance_m),
        distance_m=distance_m,
    )


def _mock_observe(hint: TargetInfo | None = None) -> ObserveFn:
    """observe_fn that always returns the same hint (or None)."""

    async def observe(arm_id: str, target_handle: str) -> TargetInfo | None:
        return (
            hint
            if hint is None
            else TargetInfo(
                handle=target_handle,
                hint_valid=hint.hint_valid,
                confidence=hint.confidence,
                obs_age_ms=hint.obs_age_ms,
                time_skew_ms=hint.time_skew_ms,
                delta_xyz_ee=hint.delta_xyz_ee,
                distance_m=hint.distance_m,
            )
        )

    return observe


async def _null_observe(arm_id: str, target_handle: str) -> None:
    return None


def _make_svc(
    rt: HALORuntime,
    observe_fn: ObserveFn = None,
    vlm_fn: VlmFn | None = None,
    cfg: TargetPerceptionServiceConfig | None = None,
    capture_fn: CaptureFn | None = None,
    tracker_factory_fn: TrackerFactoryFn | None = None,
) -> TargetPerceptionService:
    if observe_fn is None:
        observe_fn = _mock_observe(_good_hint())
    return TargetPerceptionService(
        arm_id=ARM,
        runtime=rt,
        observe_fn=observe_fn,
        vlm_fn=vlm_fn,
        capture_fn=capture_fn,
        tracker_factory_fn=tracker_factory_fn,
        config=cfg or _cfg(),
    )


# ─── tick: no target ─────────────────────────────────────────────────────────


async def test_tick_no_target_publishes_idle(rt: HALORuntime):
    svc = _make_svc(rt)
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.IDLE
    assert snap.target is None


# ─── tick: tracking ───────────────────────────────────────────────────────────


async def test_tick_with_target_publishes_tracking(rt: HALORuntime):
    svc = _make_svc(rt)
    await svc.set_tracking_target("cube-1")
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.TRACKING
    assert snap.target is not None
    assert snap.target.hint_valid is True
    assert snap.target.handle == "cube-1"


async def test_hint_stored_in_runtime_after_tick(rt: HALORuntime):
    svc = _make_svc(rt, observe_fn=_mock_observe(_good_hint(distance_m=0.25)))
    await svc.set_tracking_target("cube-1")
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.target is not None
    assert snap.target.distance_m == pytest.approx(0.25)
    assert snap.target.confidence == pytest.approx(0.95)


# ─── tick: transient loss ─────────────────────────────────────────────────────


async def test_tick_observe_none_publishes_relocalizing(rt: HALORuntime):
    svc = _make_svc(rt, observe_fn=_null_observe)
    await svc.set_tracking_target("cube-1")
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.RELOCALIZING
    assert snap.target is None


# ─── tick: reacquire failure ──────────────────────────────────────────────────


async def test_tick_reacquire_fail_limit_triggers_reacquire_failed(rt: HALORuntime):
    cfg = _cfg(reacquire_fail_limit=3)
    svc = _make_svc(rt, observe_fn=_null_observe, cfg=cfg)
    await svc.set_tracking_target("cube-1")

    for _ in range(3):
        await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.REACQUIRING
    assert snap.perception.failure_code == PerceptionFailureCode.REACQUIRE_FAILED


async def test_tick_reacquire_failed_emits_perception_failure_event(rt: HALORuntime):
    cfg = _cfg(reacquire_fail_limit=2)
    svc = _make_svc(rt, observe_fn=_null_observe, cfg=cfg)
    await svc.set_tracking_target("cube-1")

    for _ in range(2):
        await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    failure_events = [e for e in events if e.type == EventType.PERCEPTION_FAILURE]
    assert len(failure_events) == 1
    assert failure_events[0].data["failure_code"] == PerceptionFailureCode.REACQUIRE_FAILED.value


async def test_tick_failure_event_emitted_only_once_on_transition(rt: HALORuntime):
    """PERCEPTION_FAILURE fires on the first failure tick, not on every subsequent one."""
    cfg = _cfg(reacquire_fail_limit=1)
    svc = _make_svc(rt, observe_fn=_null_observe, cfg=cfg)
    await svc.set_tracking_target("cube-1")

    for _ in range(4):
        await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    failure_events = [e for e in events if e.type == EventType.PERCEPTION_FAILURE]
    assert len(failure_events) == 1


# ─── tick: recovery ───────────────────────────────────────────────────────────


async def test_tick_recovery_emits_perception_recovered_event(rt: HALORuntime):
    """Transition failure → OK emits PERCEPTION_RECOVERED."""
    cfg = _cfg(reacquire_fail_limit=1)
    call_count = 0

    async def flaky_observe(arm_id: str, target_handle: str) -> TargetInfo | None:
        nonlocal call_count
        call_count += 1
        return None if call_count == 1 else _good_hint()

    svc = _make_svc(rt, observe_fn=flaky_observe, cfg=cfg)
    await svc.set_tracking_target("cube-1")

    await svc.tick()  # → REACQUIRE_FAILED, emits PERCEPTION_FAILURE
    await svc.tick()  # → TRACKING, emits PERCEPTION_RECOVERED

    events = rt.bus.get_recent_events(ARM)
    types = [e.type for e in events]
    assert EventType.PERCEPTION_FAILURE in types
    assert EventType.PERCEPTION_RECOVERED in types


# ─── plausibility gates ───────────────────────────────────────────────────────


async def test_tick_stale_obs_age_invalidates_hint(rt: HALORuntime):
    cfg = _cfg(obs_age_limit_ms=100)
    stale_hint = TargetInfo(
        handle="cube-1",
        hint_valid=True,
        confidence=0.9,
        obs_age_ms=200,  # > 100 ms limit
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, 0.3),
        distance_m=0.3,
    )
    svc = _make_svc(rt, observe_fn=_mock_observe(stale_hint), cfg=cfg)
    await svc.set_tracking_target("cube-1")
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.target.hint_valid is False
    assert snap.perception.failure_code == PerceptionFailureCode.DEPTH_INVALID
    assert snap.perception.tracking_status == TrackingStatus.RELOCALIZING


async def test_tick_stale_time_skew_invalidates_hint(rt: HALORuntime):
    cfg = _cfg(time_skew_limit_ms=30)
    skewed_hint = TargetInfo(
        handle="cube-1",
        hint_valid=True,
        confidence=0.9,
        obs_age_ms=5,
        time_skew_ms=80,  # > 30 ms limit
        delta_xyz_ee=(0.0, 0.0, 0.3),
        distance_m=0.3,
    )
    svc = _make_svc(rt, observe_fn=_mock_observe(skewed_hint), cfg=cfg)
    await svc.set_tracking_target("cube-1")
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.target.hint_valid is False
    assert snap.perception.failure_code == PerceptionFailureCode.CALIB_INVALID


# ─── set / clear target ───────────────────────────────────────────────────────


async def test_set_tracking_target_resets_fail_count(rt: HALORuntime):
    cfg = _cfg(reacquire_fail_limit=3)
    svc = _make_svc(rt, observe_fn=_null_observe, cfg=cfg)
    await svc.set_tracking_target("cube-1")

    for _ in range(2):
        await svc.tick()

    assert svc._reacquire_fail_count == 2

    # Re-setting the target resets the count
    await svc.set_tracking_target("cube-1")
    assert svc._reacquire_fail_count == 0


async def test_clear_tracking_target_publishes_lost(rt: HALORuntime):
    svc = _make_svc(rt)
    await svc.set_tracking_target("cube-1")
    await svc.tick()  # → TRACKING

    await svc.clear_tracking_target()
    await svc.tick()  # → LOST

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.LOST
    assert snap.target is None


# ─── request_refresh ─────────────────────────────────────────────────────────


async def test_request_refresh_sets_reacquiring_status(rt: HALORuntime):
    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        await asyncio.sleep(0.05)
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=vlm)
    await svc.set_tracking_target("cube-1")
    await svc.tick()  # → TRACKING

    await svc.request_refresh()
    assert svc._tracking_status == TrackingStatus.REACQUIRING
    assert svc._vlm_job_pending is True


async def test_request_refresh_clears_after_successful_observe(rt: HALORuntime):
    svc = _make_svc(rt)
    await svc.set_tracking_target("cube-1")
    await svc.request_refresh()

    await svc.tick()  # observe succeeds → should clear vlm_job_pending

    assert svc._vlm_job_pending is False
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.TRACKING


async def test_request_refresh_without_vlm_fn_does_not_set_pending(rt: HALORuntime):
    svc = _make_svc(rt, vlm_fn=None)
    await svc.request_refresh(reason="no-vlm")

    assert svc._vlm_job_pending is False


async def test_request_refresh_scene_only_clears_pending_after_vlm_finish(rt: HALORuntime):
    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=vlm)
    await svc.request_refresh(mode="scene_only", reason="test")
    assert svc._vlm_job_pending is True

    await asyncio.sleep(0.05)

    assert svc._vlm_job_pending is False


# ─── lifecycle ────────────────────────────────────────────────────────────────


async def test_start_stop_lifecycle(rt: HALORuntime):
    svc = _make_svc(rt)
    assert svc._loop_task is None

    await svc.start()
    assert svc._loop_task is not None

    await svc.stop()
    assert svc._loop_task is None


# ─── VLM helpers ──────────────────────────────────────────────────────────────


def _scene(*handles: str) -> VlmScene:
    """Build a VlmScene with one detection per handle."""
    dets = [
        VlmDetection(
            handle=h,
            label=h,
            bbox=(0.0, 0.0, 100.0, 100.0),
            centroid=(50.0, 50.0),
            is_graspable=True,
        )
        for h in handles
    ]
    return VlmScene(scene="test scene", detections=dets)


# ─── handle stabilization ─────────────────────────────────────────────────────


def test_stabilize_scene_for_tracked_target_remaps_when_near_anchor():
    scene = VlmScene(
        scene="table",
        detections=[
            VlmDetection(
                handle="black_cube_02",
                label="cube",
                bbox=(102.0, 101.0, 201.0, 199.0),
                centroid=(151.5, 150.0),
                is_graspable=True,
            )
        ],
    )

    stabilized = _stabilize_scene_for_tracked_target(
        scene,
        tracked_handle="black_cube_01",
        tracked_center_px=(150.0, 150.0),
    )
    assert stabilized.detections[0].handle == "black_cube_01"


def test_stabilize_scene_for_tracked_target_keeps_handle_when_far():
    scene = VlmScene(
        scene="table",
        detections=[
            VlmDetection(
                handle="black_cube_02",
                label="cube",
                bbox=(700.0, 400.0, 760.0, 460.0),
                centroid=(730.0, 430.0),
                is_graspable=True,
            )
        ],
    )

    stabilized = _stabilize_scene_for_tracked_target(
        scene,
        tracked_handle="black_cube_01",
        tracked_center_px=(20.0, 20.0),
    )
    assert stabilized.detections[0].handle == "black_cube_02"


# ─── VLM async path ───────────────────────────────────────────────────────────


async def test_vlm_triggered_on_set_tracking_target(rt: HALORuntime):
    called: list[str] = []

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        called.append(arm_id)
        return _scene("cube-1")

    cfg = _cfg(tracker_init_retries=1)
    svc = _make_svc(rt, vlm_fn=vlm, cfg=cfg)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)  # yield so the VLM task can run

    assert called == [ARM]


async def test_supports_vlm_target_handle_false_for_three_arg_fn(rt: HALORuntime):
    seen: dict[str, object] = {}

    async def vlm3(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        seen["arm_id"] = arm_id
        seen["known_handles"] = list(known_handles or [])
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=vlm3)
    svc._known_handles = ["h1", "h2"]

    assert svc._supports_vlm_target_handle() is False
    out = await svc._call_vlm(vlm_image="img", target_handle="cube-1")

    assert out.detections[0].handle == "cube-1"
    assert seen["arm_id"] == ARM
    assert seen["known_handles"] == ["h1", "h2"]


async def test_supports_vlm_target_handle_true_for_four_arg_fn(rt: HALORuntime):
    seen: dict[str, object] = {}

    async def vlm4(
        arm_id: str,
        image: object = None,
        known_handles=None,
        target_handle: str | None = None,
    ) -> VlmScene:
        seen["arm_id"] = arm_id
        seen["known_handles"] = list(known_handles or [])
        seen["target_handle"] = target_handle
        return _scene(target_handle or "cube-1")

    svc = _make_svc(rt, vlm_fn=vlm4)
    svc._known_handles = ["h3"]

    assert svc._supports_vlm_target_handle() is True
    out = await svc._call_vlm(vlm_image="img", target_handle="cube-9")

    assert out.detections[0].handle == "cube-9"
    assert seen["arm_id"] == ARM
    assert seen["known_handles"] == ["h3"]
    assert seen["target_handle"] == "cube-9"


async def test_vlm_triggered_on_request_refresh(rt: HALORuntime):
    called: list[str] = []

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        called.append(arm_id)
        return _scene("cube-1")

    cfg = _cfg(tracker_init_retries=1)
    svc = _make_svc(rt, vlm_fn=vlm, cfg=cfg)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)  # drain initial VLM from set_tracking_target
    called.clear()

    await svc.request_refresh()
    await asyncio.sleep(0.05)

    assert len(called) == 1


async def test_vlm_triggered_on_request_refresh_without_target(rt: HALORuntime):
    """request_refresh works even without a tracking target (scene-only)."""
    called: list[str] = []

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        called.append(arm_id)
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=vlm)
    # No set_tracking_target — scene analysis only.
    await svc.request_refresh(reason="startup")
    await asyncio.sleep(0.05)

    assert len(called) == 1


async def test_vlm_triggered_on_reacquire_fail_limit(rt: HALORuntime):
    """When observe_fn returns None enough times, VLM is re-spawned for reacquisition."""
    call_count = 0

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        nonlocal call_count
        call_count += 1
        return _scene("cube-1")  # return matching detection (no tracker → exhausts)

    cfg = _cfg(reacquire_fail_limit=2, tracker_init_retries=1)
    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm, cfg=cfg)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)  # drain initial VLM (sets LOST because no tracker)
    call_count = 0
    # Reset from LOST so the observe path can work again
    svc._tracking_status = TrackingStatus.RELOCALIZING
    svc._reacquire_fail_count = 0

    await svc.tick()  # fail 1 — not at limit yet
    await svc.tick()  # fail 2 — hits limit, spawns VLM
    await asyncio.sleep(0.05)  # let VLM task run

    assert call_count == 1


async def test_vlm_replay_tracker_used_when_observe_returns_none(rt: HALORuntime):
    """VLM finds target → replay inits tracker → tick uses active tracker → TRACKING."""

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    capture_fn = make_mock_capture_fn()
    tracker_factory_fn = make_mock_tracker_factory_fn()

    svc = _make_svc(
        rt,
        observe_fn=_null_observe,
        vlm_fn=vlm,
        capture_fn=capture_fn,
        tracker_factory_fn=tracker_factory_fn,
    )
    await svc.set_tracking_target("cube-1")
    # tick() pushes a frame into the replay buffer; the VLM task (already
    # running) picks it up, inits the tracker, and sets _active_tracker_fn.
    await svc.tick()
    await asyncio.sleep(0.05)  # let VLM + replay task finish

    await svc.tick()  # active tracker produces hint → TRACKING

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.TRACKING
    assert snap.target is not None


async def test_vlm_reacquire_fails_without_tracker(rt: HALORuntime):
    """VLM finds target but no tracker available → retries exhaust → LOST."""

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cfg = _cfg(tracker_init_retries=1)
    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm, cfg=cfg)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)  # VLM completes, no tracker → failure

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.LOST
    assert snap.perception.failure_code == PerceptionFailureCode.REACQUIRE_FAILED


async def test_vlm_does_not_block_tick(rt: HALORuntime):
    """tick() must return promptly even when VLM is still running."""
    vlm_started = asyncio.Event()

    async def slow_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        vlm_started.set()
        await asyncio.sleep(60.0)  # intentionally very slow
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=slow_vlm)
    await svc.set_tracking_target("cube-1")
    await vlm_started.wait()  # VLM is now blocked inside sleep

    t0 = asyncio.get_event_loop().time()
    await svc.tick()  # must not wait for VLM
    elapsed = asyncio.get_event_loop().time() - t0

    assert elapsed < 0.5
    await svc.stop()  # cancels the slow VLM task


async def test_vlm_job_pending_while_task_running(rt: HALORuntime):
    vlm_proceed = asyncio.Event()

    async def blocking_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        await vlm_proceed.wait()
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=blocking_vlm)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0)  # let VLM task start

    assert svc._vlm_job_pending is True

    vlm_proceed.set()
    await asyncio.sleep(0.05)  # let VLM finish

    assert svc._vlm_job_pending is False


async def test_vlm_not_stacked(rt: HALORuntime):
    """A second trigger while VLM is running must not spawn a second task."""
    call_count = 0
    vlm_proceed = asyncio.Event()

    async def counting_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        nonlocal call_count
        call_count += 1
        await vlm_proceed.wait()
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=counting_vlm)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0)  # VLM task is now running

    # Two more triggers while the first is still in progress
    await svc.request_refresh()
    await svc.request_refresh()

    assert call_count == 1  # only the initial VLM from set_tracking_target

    vlm_proceed.set()
    await asyncio.sleep(0.05)


async def test_stop_cancels_vlm_task(rt: HALORuntime):
    cancelled = False

    async def long_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        nonlocal cancelled
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            cancelled = True
            raise
        return VlmScene(scene="", detections=[])

    svc = _make_svc(rt, vlm_fn=long_vlm)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0)  # let VLM task start

    await svc.stop()  # must cancel the VLM task and await it

    assert cancelled is True
    assert svc._vlm_task is None


async def test_vlm_mismatch_emits_failure(rt: HALORuntime):
    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("other-1")

    cfg = _cfg(tracker_init_retries=1)
    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm, cfg=cfg)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)

    events = rt.bus.get_recent_events(ARM)
    failures = [e for e in events if e.type == EventType.PERCEPTION_FAILURE]
    assert len(failures) == 1
    assert failures[0].data["failure_code"] == PerceptionFailureCode.REACQUIRE_FAILED.value

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.failure_code == PerceptionFailureCode.REACQUIRE_FAILED
    assert snap.perception.tracking_status == TrackingStatus.LOST


async def test_fuzzy_reacquire_keeps_requested_target_handle(rt: HALORuntime):
    """If fuzzy fallback picks another detection, keep the requested handle as canonical."""

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("black_cube_02")

    cap_fn = make_mock_capture_fn()
    factory_fn = make_mock_tracker_factory_fn()
    svc = _make_svc(
        rt,
        observe_fn=_null_observe,
        vlm_fn=vlm,
        capture_fn=cap_fn,
        tracker_factory_fn=factory_fn,
    )
    await svc.set_tracking_target("black_cube_01")

    await svc.tick()  # push frame for replay init
    await asyncio.sleep(0.1)  # let VLM + replay complete
    await svc.tick()  # consume seed / publish TRACKING + TARGET_ACQUIRED

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert svc._target_handle == "black_cube_01"
    assert snap.target is not None
    assert snap.target.handle == "black_cube_01"

    events = rt.bus.get_recent_events(ARM)
    acquired = [e for e in events if e.type == EventType.TARGET_ACQUIRED]
    assert acquired
    assert acquired[-1].data["target_handle"] == "black_cube_01"


# ─── TARGET_ACQUIRED event ───────────────────────────────────────────────────


async def test_target_acquired_emitted_on_first_tracking(rt: HALORuntime):
    """TARGET_ACQUIRED fires on the first tick that reaches TRACKING after set_tracking_target."""
    svc = _make_svc(rt)
    await svc.set_tracking_target("cube-1")
    await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    acquired = [e for e in events if e.type == EventType.TARGET_ACQUIRED]
    assert len(acquired) == 1
    assert acquired[0].data["target_handle"] == "cube-1"


async def test_target_acquired_not_emitted_on_subsequent_ticks(rt: HALORuntime):
    """TARGET_ACQUIRED fires only once per set_tracking_target, not on every tick."""
    svc = _make_svc(rt)
    await svc.set_tracking_target("cube-1")

    for _ in range(5):
        await svc.tick()

    events = rt.bus.get_recent_events(ARM)
    acquired = [e for e in events if e.type == EventType.TARGET_ACQUIRED]
    assert len(acquired) == 1


# ─── _drain_commands: TRACK_OBJECT ──────────────────────────────────────────


async def test_track_object_does_not_emit_scene_described(rt: HALORuntime):
    """VLM triggered by set_tracking_target must NOT emit SCENE_DESCRIBED."""

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=vlm)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)  # let VLM complete
    await svc.tick()  # consume seed → TRACKING → TARGET_ACQUIRED

    events = rt.bus.get_recent_events(ARM)
    scene_events = [e for e in events if e.type == EventType.SCENE_DESCRIBED]
    acquired_events = [e for e in events if e.type == EventType.TARGET_ACQUIRED]
    assert len(scene_events) == 0
    assert len(acquired_events) == 1


async def test_drain_commands_handles_track_object(rt: HALORuntime):
    """A COMMAND_ACCEPTED event for TRACK_OBJECT sets the tracking target."""
    svc = _make_svc(rt)
    await svc.start()
    await asyncio.sleep(0.02)

    # Simulate a COMMAND_ACCEPTED event with TRACK_OBJECT payload data
    evt = EventEnvelope(
        event_id=rt.bus.make_event_id(),
        type=EventType.COMMAND_ACCEPTED,
        ts_ms=1000,
        arm_id=ARM,
        data={
            "command_id": "cmd-track-1",
            "command_type": CommandType.TRACK_OBJECT,
            "target_handle": "mug-3",
        },
    )
    await rt.bus.publish(evt)
    await asyncio.sleep(0.1)

    assert svc._target_handle == "mug-3"

    await svc.stop()


# ─── Frame-buffer replay + switchover ────────────────────────────────────────


async def test_replay_produces_caught_up_seed(rt: HALORuntime):
    """VLM with capture+tracker replays buffer; seed has real tracker values."""
    update_hint = TargetInfo(
        handle="cube-1",
        hint_valid=True,
        confidence=0.85,
        obs_age_ms=5,
        time_skew_ms=0,
        delta_xyz_ee=(0.01, -0.02, -0.12),
        distance_m=0.12,
    )

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    factory_fn = make_mock_tracker_factory_fn(update_hint=update_hint)

    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm, capture_fn=cap_fn, tracker_factory_fn=factory_fn)
    await svc.set_tracking_target("cube-1")

    # Push a few frames into the buffer while VLM "runs"
    for _ in range(3):
        await svc.tick()

    # Let VLM + replay complete
    await asyncio.sleep(0.1)

    # After replay, _vlm_seed should have real tracker values
    assert svc._vlm_seed is not None
    assert svc._vlm_seed.distance_m == pytest.approx(0.12)
    assert svc._vlm_seed.confidence == pytest.approx(0.85)

    # Active tracker should be set (switchover)
    assert svc._active_tracker_fn is not None


async def test_switchover_tick_uses_active_tracker(rt: HALORuntime):
    """After switchover, tick() feeds frames to the active tracker, not observe_fn."""
    observe_calls: list[str] = []

    async def counting_observe(arm_id: str, target_handle: str) -> TargetInfo | None:
        observe_calls.append(target_handle)
        return _good_hint(handle=target_handle)

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    factory_fn = make_mock_tracker_factory_fn()

    svc = _make_svc(rt, observe_fn=counting_observe, vlm_fn=vlm, capture_fn=cap_fn, tracker_factory_fn=factory_fn)
    await svc.set_tracking_target("cube-1")

    # Push frames + let VLM+replay finish
    await svc.tick()
    await asyncio.sleep(0.1)

    # Clear observe_calls to track post-switchover behaviour
    observe_calls.clear()

    # Tick after switchover: should use active_tracker, not observe_fn
    await svc.tick()
    await svc.tick()

    assert len(observe_calls) == 0  # observe_fn should NOT be called
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.TRACKING


async def test_frames_captured_during_vlm_are_replayed(rt: HALORuntime):
    """Frames pushed via tick() during VLM inference are consumed by replay."""
    replayed_frames: list[CapturedFrame] = []

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        # Slow enough for a few ticks to push frames
        await asyncio.sleep(0.15)
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()

    init_hint = _good_hint(handle="cube-1")

    async def factory(frame: CapturedFrame, detection: VlmDetection):
        replayed_frames.append(frame)
        seed = dataclasses.replace(init_hint, handle=detection.handle)

        async def update(f: CapturedFrame) -> TargetInfo | None:
            replayed_frames.append(f)
            return seed

        return seed, update

    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm, capture_fn=cap_fn, tracker_factory_fn=factory)
    await svc.set_tracking_target("cube-1")

    # Push frames during VLM inference
    for _ in range(5):
        await svc.tick()
        await asyncio.sleep(0.02)

    # Wait for VLM + replay to complete
    await asyncio.sleep(0.3)

    # All frames pushed to the buffer should have been replayed
    assert len(replayed_frames) >= 3  # at least some were replayed


async def test_observe_fn_keeps_running_during_vlm(rt: HALORuntime):
    """The old observe_fn tracker keeps running while VLM inference is active."""
    observe_calls: list[str] = []
    vlm_started = asyncio.Event()

    async def counting_observe(arm_id: str, target_handle: str) -> TargetInfo | None:
        observe_calls.append(target_handle)
        return _good_hint(handle=target_handle)

    async def slow_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        vlm_started.set()
        await asyncio.sleep(60.0)
        return _scene("cube-1")

    svc = _make_svc(rt, observe_fn=counting_observe, vlm_fn=slow_vlm)
    await svc.set_tracking_target("cube-1")
    await vlm_started.wait()

    observe_calls.clear()
    await svc.tick()
    await svc.tick()
    await svc.tick()

    assert len(observe_calls) == 3  # observe_fn called every tick during VLM
    await svc.stop()


async def test_active_tracker_keeps_running_during_vlm(rt: HALORuntime):
    """If active tracker is set and a new VLM fires, the active tracker keeps being fed frames."""
    active_calls: list[str] = []

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()

    first_init = True

    async def factory(frame: CapturedFrame, detection: VlmDetection):
        nonlocal first_init
        seed = _good_hint(handle=detection.handle)

        async def update(f: CapturedFrame) -> TargetInfo | None:
            active_calls.append(f.image)
            return seed

        first_init = False
        return seed, update

    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm, capture_fn=cap_fn, tracker_factory_fn=factory)
    await svc.set_tracking_target("cube-1")
    await svc.tick()
    await asyncio.sleep(0.1)  # VLM + replay completes, switchover done

    active_calls.clear()

    # Now active tracker is set. Trigger another VLM (e.g. request_refresh).
    await svc.request_refresh()

    # Tick during the second VLM — active tracker should still be fed
    await svc.tick()
    await svc.tick()

    assert len(active_calls) >= 2
    await svc.stop()


async def test_set_tracking_target_resets_active_tracker(rt: HALORuntime):
    """set_tracking_target clears the active tracker, reverting to observe_fn."""

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    factory_fn = make_mock_tracker_factory_fn()

    svc = _make_svc(
        rt,
        observe_fn=_mock_observe(_good_hint()),
        vlm_fn=vlm,
        capture_fn=cap_fn,
        tracker_factory_fn=factory_fn,
    )
    await svc.set_tracking_target("cube-1")
    await svc.tick()
    await asyncio.sleep(0.1)  # switchover done

    assert svc._active_tracker_fn is not None

    # Re-set target — should clear active tracker
    await svc.set_tracking_target("cube-2")
    assert svc._active_tracker_fn is None


async def test_reacquire_failed_without_capture_fn(rt: HALORuntime):
    """Without capture_fn, replay fails → retries exhaust → LOST."""

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cfg = _cfg(tracker_init_retries=1)
    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm, cfg=cfg)  # no capture_fn
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.LOST
    assert snap.perception.failure_code == PerceptionFailureCode.REACQUIRE_FAILED
    assert svc._active_tracker_fn is None


async def test_reacquire_failed_without_tracker_factory(rt: HALORuntime):
    """Without tracker_factory_fn, replay fails → retries exhaust → LOST."""

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    cfg = _cfg(tracker_init_retries=1)
    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm, capture_fn=cap_fn, cfg=cfg)  # no tracker_factory
    await svc.set_tracking_target("cube-1")
    await svc.tick()  # push a frame
    await asyncio.sleep(0.1)

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.LOST
    assert snap.perception.failure_code == PerceptionFailureCode.REACQUIRE_FAILED
    assert svc._active_tracker_fn is None


async def test_buffer_starts_on_spawn_vlm_for_new_target(rt: HALORuntime):
    """Frame buffer becomes active when VLM is spawned for a new target."""
    vlm_started = asyncio.Event()

    async def slow_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        vlm_started.set()
        await asyncio.sleep(60.0)
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    svc = _make_svc(rt, vlm_fn=slow_vlm, capture_fn=cap_fn)
    await svc.set_tracking_target("cube-1")
    await vlm_started.wait()

    assert svc._replay_buffer.is_active
    await svc.stop()


async def test_buffer_not_started_on_describe_scene(rt: HALORuntime):
    """describe_scene (no new target) should NOT start frame capture."""

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    svc = _make_svc(rt, vlm_fn=vlm, capture_fn=cap_fn)
    # No set_tracking_target — just describe_scene
    await svc.request_refresh(reason="startup")

    # Buffer should not be active (for_new_target=False because no target set)
    assert not svc._replay_buffer.is_active


async def test_cancel_vlm_stops_and_clears_buffer(rt: HALORuntime):
    """Cancelling VLM clears old buffered frames; new VLM spawn may re-activate."""
    vlm_started = asyncio.Event()

    async def slow_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        vlm_started.set()
        await asyncio.sleep(60.0)
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    svc = _make_svc(rt, vlm_fn=slow_vlm, capture_fn=cap_fn)
    await svc.set_tracking_target("cube-1")
    await vlm_started.wait()

    # Push some frames
    await svc.tick()
    assert len(svc._replay_buffer) >= 1

    # Cancel via set_tracking_target (calls _cancel_vlm then _spawn_vlm)
    await svc.set_tracking_target("cube-2")
    # Old frames should be gone (cleared by _cancel_vlm); buffer may be
    # re-activated by the new _spawn_vlm for "cube-2".
    assert len(svc._replay_buffer) == 0

    await svc.stop()


async def test_cancel_vlm_without_respawn_deactivates_buffer(rt: HALORuntime):
    """clear_tracking_target calls _cancel_vlm without respawn, so buffer is inactive."""
    vlm_started = asyncio.Event()

    async def slow_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        vlm_started.set()
        await asyncio.sleep(60.0)
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    svc = _make_svc(rt, vlm_fn=slow_vlm, capture_fn=cap_fn)
    await svc.set_tracking_target("cube-1")
    await vlm_started.wait()

    await svc.tick()
    assert len(svc._replay_buffer) >= 1

    await svc.clear_tracking_target()
    assert not svc._replay_buffer.is_active
    assert len(svc._replay_buffer) == 0

    await svc.stop()


async def test_retarget_during_vlm_keeps_new_buffer_active(rt: HALORuntime):
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    calls = 0

    async def blocking_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await asyncio.sleep(60.0)
            return _scene("cube-1")
        second_started.set()
        await release_second.wait()
        return _scene("cube-2")

    cap_fn = make_mock_capture_fn()
    svc = _make_svc(rt, vlm_fn=blocking_vlm, capture_fn=cap_fn)
    await svc.set_tracking_target("cube-1")
    await first_started.wait()
    assert svc._replay_buffer.is_active

    await svc.set_tracking_target("cube-2")
    await second_started.wait()
    await asyncio.sleep(0.02)
    assert svc._replay_buffer.is_active

    release_second.set()
    await asyncio.sleep(0.05)
    await svc.stop()


async def test_replay_handles_tracker_init_failure(rt: HALORuntime):
    """If tracker_factory_fn raises on all retries → LOST."""

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()

    async def failing_factory(frame: CapturedFrame, detection: VlmDetection):
        raise RuntimeError("tracker init failed")

    cfg = _cfg(tracker_init_retries=1)
    svc = _make_svc(
        rt, observe_fn=_null_observe, vlm_fn=vlm, capture_fn=cap_fn, tracker_factory_fn=failing_factory, cfg=cfg
    )
    await svc.set_tracking_target("cube-1")
    await svc.tick()  # push a frame
    await asyncio.sleep(0.1)

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.LOST
    assert snap.perception.failure_code == PerceptionFailureCode.REACQUIRE_FAILED
    assert svc._active_tracker_fn is None


async def test_tracker_init_retries_then_succeeds(rt: HALORuntime):
    """tracker_factory fails twice, succeeds on 3rd attempt → TRACKING."""
    attempt = 0

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()

    async def flaky_factory(frame: CapturedFrame, detection: VlmDetection):
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise RuntimeError(f"fail #{attempt}")
        return await make_mock_tracker_factory_fn()(frame, detection)

    cfg = _cfg(tracker_init_retries=3)
    svc = _make_svc(
        rt, observe_fn=_null_observe, vlm_fn=vlm, capture_fn=cap_fn, tracker_factory_fn=flaky_factory, cfg=cfg
    )
    await svc.start()
    await svc.set_tracking_target("cube-1")
    # Let the service loop run: each retry re-spawns VLM, tick() captures
    # frames, replay inits the tracker.
    for _ in range(30):
        await asyncio.sleep(0.05)
        if attempt >= 3:
            break

    assert attempt == 3
    # Let the final successful replay complete
    await asyncio.sleep(0.1)
    assert svc._active_tracker_fn is not None

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.TRACKING
    await svc.stop()


async def test_tracker_init_retries_exhausted_sets_lost(rt: HALORuntime):
    """tracker_factory fails on all attempts → LOST."""
    attempt = 0

    async def vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()

    async def always_failing_factory(frame: CapturedFrame, detection: VlmDetection):
        nonlocal attempt
        attempt += 1
        raise RuntimeError(f"fail #{attempt}")

    # Use high reacquire_fail_limit so observe_fn=None ticks don't re-spawn
    # VLM before all 3 tracker init attempts are exhausted.
    cfg = _cfg(tracker_init_retries=3, reacquire_fail_limit=100)
    svc = _make_svc(
        rt, observe_fn=_null_observe, vlm_fn=vlm, capture_fn=cap_fn, tracker_factory_fn=always_failing_factory, cfg=cfg
    )
    await svc.start()
    await svc.set_tracking_target("cube-1")
    for _ in range(30):
        await asyncio.sleep(0.05)
        if attempt >= 3:
            break
    await asyncio.sleep(0.1)

    assert attempt == 3
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.LOST
    assert snap.perception.failure_code == PerceptionFailureCode.REACQUIRE_FAILED
    assert svc._active_tracker_fn is None
    # Counter should be reset after exhaustion
    assert svc._tracker_init_attempts == 0
    await svc.stop()


async def test_replay_does_not_block_tick(rt: HALORuntime):
    """tick() returns promptly even while replay is running in _run_vlm."""
    vlm_started = asyncio.Event()

    async def slow_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        vlm_started.set()
        await asyncio.sleep(60.0)
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    factory_fn = make_mock_tracker_factory_fn()

    svc = _make_svc(rt, vlm_fn=slow_vlm, capture_fn=cap_fn, tracker_factory_fn=factory_fn)
    await svc.set_tracking_target("cube-1")
    await vlm_started.wait()

    t0 = asyncio.get_event_loop().time()
    await svc.tick()
    elapsed = asyncio.get_event_loop().time() - t0

    assert elapsed < 0.5
    await svc.stop()


async def test_tick_pushes_frames_during_vlm(rt: HALORuntime):
    """Each tick during VLM inference pushes a frame into the replay buffer."""
    vlm_started = asyncio.Event()

    async def slow_vlm(arm_id: str, image: object = None, known_handles=None) -> VlmScene:
        vlm_started.set()
        await asyncio.sleep(60.0)
        return _scene("cube-1")

    cap_fn = make_mock_capture_fn()
    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=slow_vlm, capture_fn=cap_fn)
    await svc.set_tracking_target("cube-1")
    await vlm_started.wait()

    await svc.tick()
    await svc.tick()
    await svc.tick()

    assert len(svc._replay_buffer) == 3
    await svc.stop()
