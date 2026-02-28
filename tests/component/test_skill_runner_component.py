"""Component tests for SkillRunnerService — real service, mocked chunk/push with latency."""

import asyncio

from halo.contracts.enums import SkillName
from halo.contracts.events import EventType
from halo.testing.mock_fns import LatencyProfile, make_mock_chunk_fn
from halo.testing.runner import HeadlessRunner, RunnerConfig
from halo.testing.state_seeder import make_act, make_perception, make_target, seed_store

ARM = "arm0"
RUN_ID = "run-component-1"


def _runner(latency: LatencyProfile, pushed: list | None = None) -> HeadlessRunner:
    if pushed is None:
        pushed = []

    async def push_fn(chunk):
        pushed.append(chunk)

    return HeadlessRunner(
        config=RunnerConfig(
            arm_id=ARM,
            enable_planner=False,
            enable_perception=False,
            enable_skill_runner=True,
            enable_control=False,
        ),
        chunk_fn=make_mock_chunk_fn(latency),
        push_fn=push_fn,
    )


async def test_happy_path_pick(latency: LatencyProfile):
    """SkillRunner drives FSM through all phases to DONE/SUCCESS."""
    from halo.services.skill_runner_service.config import SkillRunnerConfig

    pushed = []
    runner = _runner(latency, pushed)

    # Override config for instant phase transitions
    runner.skill_runner_svc._config = SkillRunnerConfig(
        grasp_persistence_ms=0,
        close_gripper_duration_ms=0,
        verify_duration_ms=0,
        lift_duration_ms=0,
        skip_verify_grasp=True,
        no_target_tolerance_ms=99999,
        select_grasp_timeout_ms=99999,
        plan_approach_timeout_ms=99999,
        move_pregrasp_timeout_ms=99999,
        visual_align_timeout_ms=99999,
        execute_approach_timeout_ms=99999,
    )

    await runner.start()
    try:
        # Seed with close target to drive through phases quickly
        await seed_store(
            runner.runtime, ARM, target=make_target(distance_m=0.005), perception=make_perception(), act=make_act()
        )

        await runner.skill_runner_svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

        # Tick multiple times to drive through all phases
        for _ in range(20):
            await seed_store(
                runner.runtime, ARM, target=make_target(distance_m=0.005), perception=make_perception(), act=make_act()
            )
            await runner.skill_runner_svc.tick()

        await asyncio.sleep(0.02)

        # Check for SKILL_SUCCEEDED event
        succeeded = runner.recorder.events_of_type(EventType.SKILL_SUCCEEDED)
        assert len(succeeded) == 1

        # Verify phase transitions happened
        enters = runner.recorder.events_of_type(EventType.PHASE_ENTER)
        assert len(enters) >= 5  # At least several phases
    finally:
        await runner.stop()


async def test_phase_transitions_emit_events(latency: LatencyProfile):
    """Phase transitions emit PHASE_EXIT and PHASE_ENTER events."""
    runner = _runner(latency)
    await runner.start()
    try:
        await seed_store(
            runner.runtime, ARM, target=make_target(distance_m=0.1), perception=make_perception(), act=make_act()
        )
        await runner.skill_runner_svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

        await asyncio.sleep(0.02)

        # Should at least have SKILL_STARTED and PHASE_ENTER (SELECT_GRASP)
        starts = runner.recorder.events_of_type(EventType.SKILL_STARTED)
        assert len(starts) == 1

        enters = runner.recorder.events_of_type(EventType.PHASE_ENTER)
        assert len(enters) >= 1
    finally:
        await runner.stop()


async def test_timeout_triggers_failure(latency: LatencyProfile):
    """Phase timeout triggers SKILL_FAILED event."""
    from halo.services.skill_runner_service.config import SkillRunnerConfig

    runner = _runner(latency)

    # Very short timeout for MOVE_PREGRASP
    runner.skill_runner_svc._config = SkillRunnerConfig(
        move_pregrasp_timeout_ms=1,  # Immediate timeout
        no_target_tolerance_ms=99999,
        select_grasp_timeout_ms=99999,
        plan_approach_timeout_ms=99999,
    )

    await runner.start()
    try:
        # Target is far — will stay in MOVE_PREGRASP until timeout
        await seed_store(
            runner.runtime, ARM, target=make_target(distance_m=0.5), perception=make_perception(), act=make_act()
        )
        await runner.skill_runner_svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

        # Tick a few times, timeout should fire
        for _ in range(5):
            await seed_store(
                runner.runtime, ARM, target=make_target(distance_m=0.5), perception=make_perception(), act=make_act()
            )
            await runner.skill_runner_svc.tick()
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.02)

        failed = runner.recorder.events_of_type(EventType.SKILL_FAILED)
        assert len(failed) >= 1
    finally:
        await runner.stop()


async def test_abort_skill(latency: LatencyProfile):
    """abort_skill() publishes SKILL_FAILED event."""
    runner = _runner(latency)
    await runner.start()
    try:
        await seed_store(
            runner.runtime, ARM, target=make_target(distance_m=0.5), perception=make_perception(), act=make_act()
        )
        await runner.skill_runner_svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
        await runner.skill_runner_svc.abort_skill()

        await asyncio.sleep(0.02)

        failed = runner.recorder.events_of_type(EventType.SKILL_FAILED)
        assert len(failed) == 1
    finally:
        await runner.stop()
