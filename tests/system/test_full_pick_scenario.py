"""System test: full pick scenario — all services wired, external deps mocked.

Scenario: planner sees scene → issues start_skill(TRACK) → perception acquires
target → planner issues start_skill(PICK) → FSM runs through phases → skill succeeds.
"""

import asyncio

from halo.contracts.commands import (
    CommandEnvelope,
    DescribeScenePayload,
    StartSkillPayload,
)
from halo.contracts.enums import CommandType, SkillName
from halo.contracts.events import EventType
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.testing.mock_fns import (
    LatencyProfile,
    make_command,
    make_mock_apply_fn,
    make_mock_capture_fn_with_latency,
    make_mock_chunk_fn,
    make_mock_tracker_factory_fn_with_latency,
    make_mock_vlm_fn,
)
from halo.testing.runner import HeadlessRunner, RunnerConfig

ARM = "arm0"


def _make_scripted_decide(latency: LatencyProfile):
    """Scripted decide_fn that drives a full pick scenario.

    Step 0: DESCRIBE_SCENE
    Step 1: START_SKILL(TRACK, red_cube)
    Step 2: wait for TRACK to succeed, then START_SKILL(PICK)
    After that: no-op until skill completes
    """
    state = {"step": 0, "skill_started": False}

    async def decide_fn(snap: PlannerSnapshot) -> list[CommandEnvelope]:
        await asyncio.sleep(latency.decide_s[0])  # minimal latency
        step = state["step"]

        if step == 0:
            state["step"] = 1
            return [make_command(ARM, CommandType.DESCRIBE_SCENE, DescribeScenePayload(reason="startup"))]

        if step == 1:
            state["step"] = 2
            return [
                make_command(
                    ARM,
                    CommandType.START_SKILL,
                    StartSkillPayload(skill_name=SkillName.TRACK, target_handle="red_cube"),
                    snapshot_id=None,
                )
            ]

        # Wait for TRACK skill to complete (DONE phase) before starting PICK
        track_done = snap.skill is not None and snap.skill.name == SkillName.TRACK and snap.skill.phase.name == "DONE"
        no_skill = snap.skill is None
        ready = snap.target is not None and snap.target.hint_valid and (track_done or no_skill)
        if not state["skill_started"] and ready:
            state["skill_started"] = True
            # In a system test with concurrent service ticks, snapshot_id can
            # become stale between decide_fn return and submit_command. Use None
            # to skip precondition check — the real PlannerAgent handles retries.
            return [
                make_command(
                    ARM,
                    CommandType.START_SKILL,
                    StartSkillPayload(skill_name=SkillName.PICK, target_handle="red_cube"),
                    snapshot_id=None,
                )
            ]

        return []

    return decide_fn


async def test_full_pick_scenario(latency: LatencyProfile):
    """Planner drives perception → tracking → pick skill → success."""
    applied = []

    # Use lenient thresholds and instant timers so the mock tracker (distance=0.15m)
    # can drive through all phases without needing decreasing distance.
    skill_cfg = SkillRunnerConfig(
        approach_align_threshold_m=0.2,  # mock tracker distance=0.15 passes this
        execute_approach_threshold_m=0.2,
        grasp_distance_threshold_m=0.2,
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

    config = RunnerConfig(
        arm_id=ARM,
        max_duration_s=10.0,
        skill_runner_config=skill_cfg,
    )

    runner = HeadlessRunner(
        config=config,
        decide_fn=_make_scripted_decide(latency),
        vlm_fn=make_mock_vlm_fn(latency),
        capture_fn=make_mock_capture_fn_with_latency(latency),
        tracker_factory_fn=make_mock_tracker_factory_fn_with_latency(latency),
        chunk_fn=make_mock_chunk_fn(latency),
        apply_fn=make_mock_apply_fn(latency, applied),
    )

    def done():
        succeeded = runner.recorder.events_of_type(EventType.SKILL_SUCCEEDED)
        return len(succeeded) > 0

    await runner.run(until=done)

    # Verify the event sequence
    succeeded = runner.recorder.events_of_type(EventType.SKILL_SUCCEEDED)
    assert len(succeeded) == 1, (
        f"Expected 1 SKILL_SUCCEEDED, got {len(succeeded)}. Events: {runner.recorder.event_types()}"
    )

    # Should have had COMMAND_ACCEPTED events (describe_scene, start_skill(TRACK), start_skill(PICK), + planner startup)
    accepted = runner.recorder.events_of_type(EventType.COMMAND_ACCEPTED)
    assert len(accepted) >= 3

    # Should have had phase transitions
    enters = runner.recorder.events_of_type(EventType.PHASE_ENTER)
    assert len(enters) >= 3  # At least SELECT_GRASP, PLAN_APPROACH, MOVE_PREGRASP...

    # Should have applied some actions
    assert len(applied) > 0
