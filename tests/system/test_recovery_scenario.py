"""System test: target loss → reacquire → resume.

Scenario: skill running → perception loses target → observe returns None →
PERCEPTION_FAILURE event → VLM reacquire → target recovered → skill resumes.
"""

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
from halo.testing.state_seeder import make_target

ARM = "arm0"


async def test_target_loss_and_recovery(latency: LatencyProfile):
    """Perception detects target loss, reacquires via VLM, and recovers."""
    runner = HeadlessRunner(
        config=RunnerConfig(
            arm_id=ARM,
            max_duration_s=5.0,
            enable_planner=False,
            enable_skill_runner=False,
            enable_control=False,
            enable_perception=True,
        ),
        vlm_fn=make_mock_vlm_fn(latency),
        capture_fn=make_mock_capture_fn_with_latency(latency),
        tracker_factory_fn=make_mock_tracker_factory_fn_with_latency(latency),
    )

    await runner.start()
    try:
        # Start tracking
        await runner.perception_svc.set_tracking_target("red_cube")

        # Tick until tracker initialises
        for _ in range(20):
            await runner.perception_svc.tick()
            await asyncio.sleep(0.02)

        acquired = runner.recorder.events_of_type(EventType.TARGET_ACQUIRED)
        assert len(acquired) >= 1, f"Target not acquired. Events: {runner.recorder.event_types()}"

        # Now clear and re-acquire to simulate loss and recovery
        runner.recorder.clear()
        await runner.perception_svc.request_refresh(reason="target_lost")

        # Tick until recovery
        for _ in range(30):
            await runner.perception_svc.tick()
            await asyncio.sleep(0.02)

        # After reacquire, target should be valid again
        snap = await runner.runtime.get_latest_runtime_snapshot(ARM)
        assert snap.target is not None
        assert snap.target.handle == "red_cube"
    finally:
        await runner.stop()


async def test_perception_failure_event_on_lost_target(latency: LatencyProfile):
    """Perception emits PERCEPTION_FAILURE when observe returns None repeatedly.

    Uses observe_fn only (no tracker) so the observe_fn failure path is
    exercised directly.  The default ``reacquire_fail_limit`` is 3, so
    after 3 consecutive ``None`` returns from observe_fn the service
    transitions to REACQUIRE_FAILED and emits PERCEPTION_FAILURE.
    """
    from halo.services.target_perception_service.config import TargetPerceptionServiceConfig

    observe_count = 0

    # Custom observe_fn that returns None after a few calls
    async def flaky_observe(arm_id: str, handle: str):
        nonlocal observe_count
        observe_count += 1
        if observe_count <= 2:
            return make_target(handle=handle, distance_m=0.1)
        return None  # simulate loss

    # No capture_fn / tracker_factory_fn — forces the service to stay on
    # observe_fn rather than switching to a tracker after VLM completes.
    runner = HeadlessRunner(
        config=RunnerConfig(
            arm_id=ARM,
            max_duration_s=5.0,
            enable_planner=False,
            enable_skill_runner=False,
            enable_control=False,
            enable_perception=True,
            perception_config=TargetPerceptionServiceConfig(reacquire_fail_limit=3),
        ),
        observe_fn=flaky_observe,
        vlm_fn=make_mock_vlm_fn(latency),
    )

    await runner.start()
    try:
        await runner.perception_svc.set_tracking_target("obj-1")

        # Tick enough times for observe to succeed twice then fail ≥3 times
        for _ in range(20):
            await runner.perception_svc.tick()
            await asyncio.sleep(0.01)

        # Must have emitted PERCEPTION_FAILURE after 3 consecutive None returns
        failures = runner.recorder.events_of_type(EventType.PERCEPTION_FAILURE)
        assert len(failures) >= 1, (
            f"Expected PERCEPTION_FAILURE event but got none. Events: {runner.recorder.event_types()}"
        )

        snap = await runner.runtime.get_latest_runtime_snapshot(ARM)
        assert snap.perception.tracking_status in (
            TrackingStatus.REACQUIRING,
            TrackingStatus.LOST,
        )
    finally:
        await runner.stop()
