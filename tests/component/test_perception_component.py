"""Component tests for TargetPerceptionService — real service, mocked VLM/tracker with latency."""

import asyncio

from halo.contracts.enums import TrackingStatus
from halo.contracts.events import EventType
from halo.testing.mock_fns import (
    LatencyProfile,
    make_mock_capture_fn_with_latency,
    make_mock_tracker_factory_fn_with_latency,
    make_mock_vlm_fn,
)
from halo.testing.runner import HeadlessRunner, RunnerConfig

ARM = "arm0"


def _runner(latency: LatencyProfile, **kwargs) -> HeadlessRunner:
    return HeadlessRunner(
        config=RunnerConfig(
            arm_id=ARM,
            enable_planner=False,
            enable_perception=True,
            enable_skill_runner=False,
            enable_control=False,
        ),
        vlm_fn=kwargs.get("vlm_fn", make_mock_vlm_fn(latency)),
        capture_fn=kwargs.get("capture_fn", make_mock_capture_fn_with_latency(latency)),
        tracker_factory_fn=kwargs.get("tracker_factory_fn", make_mock_tracker_factory_fn_with_latency(latency)),
    )


async def test_scene_describe(latency: LatencyProfile):
    """request_refresh triggers VLM and emits SCENE_DESCRIBED event."""
    runner = _runner(latency)
    await runner.start()
    try:
        await runner.perception_svc.request_refresh(reason="test")

        # Give VLM time to run
        for _ in range(20):
            await runner.perception_svc.tick()
            await asyncio.sleep(0.02)

        described = runner.recorder.events_of_type(EventType.SCENE_DESCRIBED)
        assert len(described) >= 1
    finally:
        await runner.stop()


async def test_track_object_acquires_target(latency: LatencyProfile):
    """set_tracking_target triggers VLM → tracker init → TARGET_ACQUIRED."""
    runner = _runner(latency)
    await runner.start()
    try:
        await runner.perception_svc.set_tracking_target("red_cube")

        # Tick until tracker initialises
        for _ in range(30):
            await runner.perception_svc.tick()
            await asyncio.sleep(0.02)

        acquired = runner.recorder.events_of_type(EventType.TARGET_ACQUIRED)
        assert len(acquired) >= 1

        snap = await runner.runtime.get_latest_runtime_snapshot(ARM)
        assert snap.target is not None
        assert snap.target.handle == "red_cube"
    finally:
        await runner.stop()


async def test_clear_target_stops_tracking(latency: LatencyProfile):
    """clear_tracking_target stops tracking and publishes LOST status."""
    runner = _runner(latency)
    await runner.start()
    try:
        await runner.perception_svc.set_tracking_target("red_cube")

        for _ in range(15):
            await runner.perception_svc.tick()
            await asyncio.sleep(0.02)

        await runner.perception_svc.clear_tracking_target()
        await runner.perception_svc.tick()

        snap = await runner.runtime.get_latest_runtime_snapshot(ARM)
        assert snap.perception.tracking_status in (TrackingStatus.IDLE, TrackingStatus.LOST)
    finally:
        await runner.stop()


async def test_vlm_never_blocks_tick(latency: LatencyProfile):
    """The VLM callback runs async — tick() should not be blocked by VLM latency."""
    # Use realistic latency for VLM (1-3s)
    slow_latency = LatencyProfile(vlm_s=(0.5, 1.0), capture_s=(0.001, 0.002))
    runner = _runner(slow_latency)
    await runner.start()
    try:
        await runner.perception_svc.set_tracking_target("red_cube")

        # Tick should be fast even with slow VLM
        import time

        t0 = time.monotonic()
        for _ in range(5):
            await runner.perception_svc.tick()
        elapsed = time.monotonic() - t0

        # 5 ticks should be much faster than 1 VLM call
        assert elapsed < 0.5
    finally:
        await runner.stop()
