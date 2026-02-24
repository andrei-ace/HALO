"""Tests for TargetPerceptionService: tick, lifecycle, plausibility gates, events."""

import asyncio

import pytest

from halo.contracts.enums import CommandType, PerceptionFailureCode, TrackingStatus
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import TargetInfo
from halo.runtime.runtime import HALORuntime
from halo.services.target_perception_service.config import TargetPerceptionServiceConfig
from halo.services.target_perception_service.service import (
    ObserveFn,
    TargetPerceptionService,
    VlmFn,
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
) -> TargetPerceptionService:
    if observe_fn is None:
        observe_fn = _mock_observe(_good_hint())
    return TargetPerceptionService(
        arm_id=ARM,
        runtime=rt,
        observe_fn=observe_fn,
        vlm_fn=vlm_fn,
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
    svc = _make_svc(rt)
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


# ─── VLM async path ───────────────────────────────────────────────────────────


async def test_vlm_triggered_on_set_tracking_target(rt: HALORuntime):
    called: list[str] = []

    async def vlm(arm_id: str) -> VlmScene:
        called.append(arm_id)
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=vlm)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)  # yield so the VLM task can run

    assert called == [ARM]


async def test_vlm_triggered_on_request_refresh(rt: HALORuntime):
    called: list[str] = []

    async def vlm(arm_id: str) -> VlmScene:
        called.append(arm_id)
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=vlm)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)  # drain initial VLM from set_tracking_target
    called.clear()

    await svc.request_refresh()
    await asyncio.sleep(0.05)

    assert len(called) == 1


async def test_vlm_triggered_on_request_refresh_without_target(rt: HALORuntime):
    """request_refresh works even without a tracking target (scene-only)."""
    called: list[str] = []

    async def vlm(arm_id: str) -> VlmScene:
        called.append(arm_id)
        return _scene("cube-1")

    svc = _make_svc(rt, vlm_fn=vlm)
    # No set_tracking_target — scene analysis only.
    await svc.request_refresh(reason="startup")
    await asyncio.sleep(0.05)

    assert len(called) == 1


async def test_vlm_triggered_on_reacquire_fail_limit(rt: HALORuntime):
    call_count = 0

    async def vlm(arm_id: str) -> VlmScene:
        nonlocal call_count
        call_count += 1
        return VlmScene(scene="empty", detections=[])  # no match

    cfg = _cfg(reacquire_fail_limit=2)
    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm, cfg=cfg)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)  # drain initial VLM
    call_count = 0

    await svc.tick()  # fail 1 — not at limit yet
    await svc.tick()  # fail 2 — hits limit, spawns VLM
    await asyncio.sleep(0.05)  # let VLM task run

    assert call_count == 1


async def test_vlm_seed_used_when_observe_returns_none(rt: HALORuntime):
    async def vlm(arm_id: str) -> VlmScene:
        return _scene("cube-1")

    svc = _make_svc(rt, observe_fn=_null_observe, vlm_fn=vlm)
    await svc.set_tracking_target("cube-1")
    await asyncio.sleep(0.05)  # VLM completes and places seed

    await svc.tick()  # observe=None but seed available → TRACKING

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.TRACKING
    assert snap.target is not None


async def test_vlm_does_not_block_tick(rt: HALORuntime):
    """tick() must return promptly even when VLM is still running."""
    vlm_started = asyncio.Event()

    async def slow_vlm(arm_id: str) -> VlmScene:
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

    async def blocking_vlm(arm_id: str) -> VlmScene:
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

    async def counting_vlm(arm_id: str) -> VlmScene:
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

    async def long_vlm(arm_id: str) -> VlmScene:
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

    async def vlm(arm_id: str) -> VlmScene:
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
