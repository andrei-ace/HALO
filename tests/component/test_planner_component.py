"""Component tests for PlannerService — real service, mocked decide_fn with latency."""

import asyncio
import time
import uuid

from halo.contracts.commands import CommandEnvelope, DescribeScenePayload
from halo.contracts.enums import CommandType
from halo.contracts.events import EventType
from halo.testing.mock_fns import LatencyProfile, make_mock_decide_fn
from halo.testing.runner import HeadlessRunner, RunnerConfig

ARM = "arm0"


def _runner(latency: LatencyProfile, decide_fn=None) -> HeadlessRunner:
    return HeadlessRunner(
        config=RunnerConfig(
            arm_id=ARM,
            enable_planner=True,
            enable_perception=False,
            enable_skill_runner=False,
            enable_control=False,
        ),
        decide_fn=decide_fn or make_mock_decide_fn(latency),
    )


async def test_startup_issues_describe_scene(latency: LatencyProfile):
    """PlannerService start() issues a DESCRIBE_SCENE command."""
    runner = _runner(latency)
    await runner.start()
    try:
        # Planner start() issues initial DESCRIBE_SCENE
        await asyncio.sleep(0.1)

        accepted = runner.recorder.events_of_type(EventType.COMMAND_ACCEPTED)
        assert len(accepted) >= 1
        # The first accepted command should be DESCRIBE_SCENE
        first_data = accepted[0].event.data
        assert first_data.get("command_type") == CommandType.DESCRIBE_SCENE
    finally:
        await runner.stop()


async def test_tick_calls_decide_fn(latency: LatencyProfile):
    """PlannerService.tick() calls decide_fn and submits returned commands."""
    calls = []

    async def tracking_decide(snap):
        calls.append(snap)
        return []

    runner = _runner(latency, decide_fn=tracking_decide)
    await runner.start()
    try:
        await runner.planner_svc.tick()
        assert len(calls) == 1
    finally:
        await runner.stop()


async def test_decide_fn_commands_are_submitted(latency: LatencyProfile):
    """Commands returned by decide_fn are submitted and acknowledged."""

    def commands_fn(snap):
        return [
            CommandEnvelope(
                command_id=str(uuid.uuid4()),
                arm_id=ARM,
                issued_at_ms=int(time.time() * 1000),
                type=CommandType.DESCRIBE_SCENE,
                payload=DescribeScenePayload(reason="test"),
                precondition_snapshot_id=None,
            )
        ]

    runner = _runner(latency, decide_fn=make_mock_decide_fn(latency, commands_fn=commands_fn))
    await runner.start()
    try:
        acks = await runner.planner_svc.tick()
        assert len(acks) == 1
    finally:
        await runner.stop()


async def test_urgent_event_wakes_planner(latency: LatencyProfile):
    """Publishing an urgent event wakes the planner from its watchdog sleep."""
    tick_count = []

    async def counting_decide(snap):
        tick_count.append(1)
        return []

    runner = _runner(latency, decide_fn=counting_decide)
    # Use long watchdog so only urgent events trigger ticks
    runner.planner_svc._config.watchdog_interval_s = 60.0
    await runner.start()
    try:
        initial = len(tick_count)
        await asyncio.sleep(0.05)

        # Publish an urgent event
        from halo.contracts.events import EventEnvelope, EventType

        evt = EventEnvelope(
            event_id=runner.runtime.bus.make_event_id(),
            type=EventType.SKILL_FAILED,
            ts_ms=int(time.time() * 1000),
            arm_id=ARM,
            data={"reason": "test"},
        )
        await runner.runtime.bus.publish(evt)
        await asyncio.sleep(0.2)

        # Should have ticked at least once more after the event
        assert len(tick_count) > initial
    finally:
        await runner.stop()
