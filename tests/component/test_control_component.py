"""Component tests for ControlService — real service, mocked apply_fn with latency."""

import asyncio

from halo.contracts.actions import ZERO_JOINT_ACTION, JointPositionAction, JointPositionChunk
from halo.contracts.enums import ActStatus, PhaseId
from halo.contracts.events import EventType
from halo.testing.mock_fns import LatencyProfile, make_mock_apply_fn
from halo.testing.runner import HeadlessRunner, RunnerConfig
from halo.testing.state_seeder import make_perception, make_target, seed_store

ARM = "arm0"


def _runner(latency: LatencyProfile, applied: list) -> HeadlessRunner:
    return HeadlessRunner(
        config=RunnerConfig(
            arm_id=ARM,
            enable_planner=False,
            enable_perception=False,
            enable_skill_runner=False,
            enable_control=True,
        ),
        apply_fn=make_mock_apply_fn(latency, applied),
        initial_joint_state=ZERO_JOINT_ACTION,
    )


async def test_action_streaming(latency: LatencyProfile):
    """ControlService streams actions from pushed chunks to apply_fn."""
    applied = []
    runner = _runner(latency, applied)
    await runner.start()
    try:
        # Seed state with valid target
        await seed_store(runner.runtime, ARM, target=make_target(), perception=make_perception())

        # Push a chunk and tick several times
        actions = tuple(JointPositionAction(values=(0.001, 0.0, 0.0, 0.0, 0.0, 0.5)) for _ in range(5))
        chunk = JointPositionChunk(chunk_id="c1", arm_id=ARM, phase_id=PhaseId.MOVE_PREGRASP, actions=actions, ts_ms=0)
        await runner.control_svc.push_chunk(chunk)

        for _ in range(5):
            await runner.control_svc.tick()

        assert len(applied) == 5
    finally:
        await runner.stop()


async def test_safety_reflex(latency: LatencyProfile):
    """ControlService triggers safety reflex on out-of-range joint action."""
    applied = []
    runner = _runner(latency, applied)
    await runner.start()
    try:
        await seed_store(runner.runtime, ARM, target=make_target(), perception=make_perception())

        # Push a chunk with out-of-range action (shoulder_pan=3.0 >> ±1.92 limit)
        bad_action = JointPositionAction(values=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        chunk = JointPositionChunk(
            chunk_id="bad", arm_id=ARM, phase_id=PhaseId.MOVE_PREGRASP, actions=(bad_action,), ts_ms=0
        )
        await runner.control_svc.push_chunk(chunk)
        await runner.control_svc.tick()

        await asyncio.sleep(0.02)
        reflex_events = runner.recorder.events_of_type(EventType.SAFETY_REFLEX_TRIGGERED)
        assert len(reflex_events) == 1
    finally:
        await runner.stop()


async def test_stale_hint_holds_position(latency: LatencyProfile):
    """ControlService holds last-applied position when target hint is stale."""
    applied = []
    runner = _runner(latency, applied)
    await runner.start()
    try:
        # Seed with stale hint
        await seed_store(
            runner.runtime, ARM, target=make_target(obs_age_ms=999, hint_valid=False), perception=make_perception()
        )

        actions = tuple(JointPositionAction(values=(0.001, 0.0, 0.0, 0.0, 0.0, 0.5)) for _ in range(3))
        chunk = JointPositionChunk(chunk_id="c1", arm_id=ARM, phase_id=PhaseId.MOVE_PREGRASP, actions=actions, ts_ms=0)
        await runner.control_svc.push_chunk(chunk)
        await runner.control_svc.tick()

        # Hold command sent using initial_state (ZERO_JOINT_ACTION)
        assert len(applied) >= 1
        assert all(v == 0.0 for v in applied[-1][1].values)

        snap = await runner.runtime.get_latest_runtime_snapshot(ARM)
        assert snap.act.status == ActStatus.STALE
    finally:
        await runner.stop()


async def test_buffer_underrun_status(latency: LatencyProfile):
    """ControlService reports IDLE when buffer is empty."""
    applied = []
    runner = _runner(latency, applied)
    await runner.start()
    try:
        await runner.control_svc.tick()
        snap = await runner.runtime.get_latest_runtime_snapshot(ARM)
        assert snap.act.status == ActStatus.IDLE
    finally:
        await runner.stop()
