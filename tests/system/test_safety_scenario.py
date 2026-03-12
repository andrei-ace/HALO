"""System test: safety reflex during active skill → recovery.

Scenario: skill running → control receives out-of-range chunk → safety reflex
fires → SAFETY_REFLEX_TRIGGERED event → subsequent clean tick → SAFETY_RECOVERED.
"""

import asyncio

from halo.contracts.actions import ZERO_JOINT_ACTION, JointPositionAction, JointPositionChunk
from halo.contracts.enums import SafetyState, SkillName
from halo.contracts.events import EventType
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.testing.mock_fns import (
    LatencyProfile,
    make_mock_apply_fn,
)
from halo.testing.runner import HeadlessRunner, RunnerConfig
from halo.testing.state_seeder import make_act, make_perception, make_target, seed_store

ARM = "arm0"


async def test_safety_reflex_and_recovery(latency: LatencyProfile):
    """Safety reflex fires on out-of-range action, then recovers on clean tick."""
    applied = []

    # Custom chunk_fn that produces one out-of-range chunk then normal chunks
    seq = 0

    async def danger_then_safe_chunk_fn(arm_id, phase, snap):
        nonlocal seq
        seq += 1
        if seq == 1:
            # Out-of-range action (shoulder_pan=3.0 >> ±1.92 limit)
            bad = JointPositionAction(values=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            return JointPositionChunk(chunk_id=f"chunk-{seq}", arm_id=arm_id, phase_id=phase, actions=(bad,), ts_ms=0)
        # Normal safe action
        safe = JointPositionAction(values=(0.001, 0.0, 0.0, 0.0, 0.0, 0.5))
        actions = tuple(safe for _ in range(10))
        return JointPositionChunk(chunk_id=f"chunk-{seq}", arm_id=arm_id, phase_id=phase, actions=actions, ts_ms=0)

    skill_cfg = SkillRunnerConfig(
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
        max_duration_s=5.0,
        enable_planner=False,
        enable_perception=False,
        skill_runner_config=skill_cfg,
    )

    runner = HeadlessRunner(
        config=config,
        chunk_fn=danger_then_safe_chunk_fn,
        apply_fn=make_mock_apply_fn(latency, applied),
        initial_joint_state=ZERO_JOINT_ACTION,
    )

    await runner.start()
    try:
        # Seed state
        await seed_store(
            runner.runtime, ARM, target=make_target(distance_m=0.005), perception=make_perception(), act=make_act()
        )

        # Start skill
        await runner.skill_runner_svc.start_skill(SkillName.PICK, "run-safety", "obj-1")

        # First tick: skill runner generates dangerous chunk → push to control
        await seed_store(
            runner.runtime, ARM, target=make_target(distance_m=0.005), perception=make_perception(), act=make_act()
        )
        await runner.skill_runner_svc.tick()

        # Control ticks to process the dangerous chunk
        for _ in range(3):
            await runner.control_svc.tick()
        await asyncio.sleep(0.02)

        reflex_events = runner.recorder.events_of_type(EventType.SAFETY_REFLEX_TRIGGERED)
        assert len(reflex_events) == 1

        # Second skill tick: generates safe chunk
        await seed_store(
            runner.runtime, ARM, target=make_target(distance_m=0.005), perception=make_perception(), act=make_act()
        )
        await runner.skill_runner_svc.tick()

        # Control ticks to process safe chunk → recovery
        for _ in range(5):
            await runner.control_svc.tick()
        await asyncio.sleep(0.02)

        recovered = runner.recorder.events_of_type(EventType.SAFETY_RECOVERED)
        assert len(recovered) == 1

        snap = await runner.runtime.get_latest_runtime_snapshot(ARM)
        assert snap.safety.state == SafetyState.OK
    finally:
        await runner.stop()
