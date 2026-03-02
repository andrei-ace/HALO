"""Component tests for SkillRunnerService in teacher mode — real service, mocked teacher_step_fn."""

import asyncio

from halo.contracts.enums import PhaseId, SkillName
from halo.contracts.events import EventType
from halo.testing.mock_fns import LatencyProfile, make_mock_teacher_step_fn
from halo.testing.runner import HeadlessRunner, RunnerConfig

ARM = "arm0"
RUN_ID = "run-teacher-1"


def _runner(latency: LatencyProfile) -> HeadlessRunner:
    return HeadlessRunner(
        config=RunnerConfig(
            arm_id=ARM,
            enable_planner=False,
            enable_perception=False,
            enable_skill_runner=True,
            enable_control=False,
        ),
        teacher_step_fn=make_mock_teacher_step_fn(latency),
    )


async def test_teacher_happy_path_pick(latency: LatencyProfile):
    """Teacher drives SkillRunner through all phases to SKILL_SUCCEEDED."""
    runner = _runner(latency)
    await runner.start()
    try:
        await runner.skill_runner_svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")

        # Tick enough times to exhaust the default phase sequence
        for _ in range(15):
            await runner.skill_runner_svc.tick()
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.02)

        succeeded = runner.recorder.events_of_type(EventType.SKILL_SUCCEEDED)
        assert len(succeeded) == 1

        enters = runner.recorder.events_of_type(EventType.PHASE_ENTER)
        assert len(enters) >= 4  # Several phase transitions
    finally:
        await runner.stop()


async def test_teacher_abort(latency: LatencyProfile):
    """abort_skill() in teacher mode publishes SKILL_FAILED."""
    # Use a phase sequence that never reaches DONE
    runner = HeadlessRunner(
        config=RunnerConfig(
            arm_id=ARM,
            enable_planner=False,
            enable_perception=False,
            enable_skill_runner=True,
            enable_control=False,
        ),
        teacher_step_fn=make_mock_teacher_step_fn(
            latency,
            phase_sequence=[(int(PhaseId.MOVE_PREGRASP), False)] * 100,
        ),
    )
    await runner.start()
    try:
        await runner.skill_runner_svc.start_skill(SkillName.PICK, RUN_ID, "obj-1")
        await runner.skill_runner_svc.tick()
        await runner.skill_runner_svc.abort_skill()

        await asyncio.sleep(0.02)

        failed = runner.recorder.events_of_type(EventType.SKILL_FAILED)
        assert len(failed) == 1
    finally:
        await runner.stop()
