"""Tests for TargetPerceptionService: tick, lifecycle, plausibility gates, events."""

import pytest

from halo.contracts.enums import PerceptionFailureCode, TrackingStatus
from halo.contracts.events import EventType
from halo.contracts.snapshots import TargetInfo
from halo.runtime.runtime import HALORuntime
from halo.services.target_perception_service.config import TargetPerceptionServiceConfig
from halo.services.target_perception_service.service import (
    ObserveFn,
    TargetPerceptionService,
)

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
        return hint if hint is None else TargetInfo(
            handle=target_handle,
            hint_valid=hint.hint_valid,
            confidence=hint.confidence,
            obs_age_ms=hint.obs_age_ms,
            time_skew_ms=hint.time_skew_ms,
            delta_xyz_ee=hint.delta_xyz_ee,
            distance_m=hint.distance_m,
        )
    return observe


async def _null_observe(arm_id: str, target_handle: str) -> None:
    return None


def _make_svc(
    rt: HALORuntime,
    observe_fn: ObserveFn = None,
    cfg: TargetPerceptionServiceConfig | None = None,
) -> TargetPerceptionService:
    if observe_fn is None:
        observe_fn = _mock_observe(_good_hint())
    return TargetPerceptionService(
        arm_id=ARM,
        runtime=rt,
        observe_fn=observe_fn,
        config=cfg or _cfg(),
    )


# ─── tick: no target ─────────────────────────────────────────────────────────

async def test_tick_no_target_publishes_lost(rt: HALORuntime):
    svc = _make_svc(rt)
    await svc.tick()

    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert snap.perception.tracking_status == TrackingStatus.LOST
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
