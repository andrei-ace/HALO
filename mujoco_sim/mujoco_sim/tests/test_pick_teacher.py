"""Tests for PickTeacher scripted policy.

No robosuite dependency — uses synthetic observation dicts.
"""

from __future__ import annotations

import numpy as np

from mujoco_sim.constants import (
    ACTION_DIM,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    PHASE_CLOSE_GRIPPER,
    PHASE_DONE,
    PHASE_EXECUTE_APPROACH,
    PHASE_IDLE,
    PHASE_LIFT,
    PHASE_MOVE_PREGRASP,
    PHASE_VERIFY_GRASP,
    PHASE_VISUAL_ALIGN,
)
from mujoco_sim.teacher.pick_teacher import PickTeacher, TeacherConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(
    ee_pos: tuple[float, float, float] = (0.5, 0.0, 0.5),
    cube_pos: tuple[float, float, float] = (0.5, 0.0, 0.1),
    gripper: float = 0.0,
) -> dict:
    """Create a minimal observation dict for teacher testing."""
    ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
    cube_quat = np.array([1.0, 0.0, 0.0, 0.0])
    return {
        "ee_pose": np.concatenate([np.array(ee_pos), ee_quat]),
        "object_pose": np.concatenate([np.array(cube_pos), cube_quat]),
        "gripper": gripper,
    }


# ---------------------------------------------------------------------------
# Init and reset
# ---------------------------------------------------------------------------


class TestPickTeacherInit:
    def test_initial_phase_is_idle(self):
        teacher = PickTeacher()
        assert teacher.phase == PHASE_IDLE
        assert teacher.done is False

    def test_reset_returns_to_idle(self):
        teacher = PickTeacher()
        obs = _make_obs(ee_pos=(0.5, 0.0, 0.11), cube_pos=(0.5, 0.0, 0.1))
        teacher.step(obs)  # advance past IDLE
        assert teacher.phase != PHASE_IDLE
        teacher.reset()
        assert teacher.phase == PHASE_IDLE

    def test_custom_config(self):
        cfg = TeacherConfig(approach_gain=1.0, max_delta=0.1)
        teacher = PickTeacher(config=cfg)
        assert teacher._config.approach_gain == 1.0


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------


class TestPhaseTransitions:
    def test_idle_to_move_pregrasp_immediate(self):
        """IDLE → SELECT_GRASP → PLAN_APPROACH → MOVE_PREGRASP on first step."""
        teacher = PickTeacher()
        obs = _make_obs(ee_pos=(0.5, 0.0, 0.5), cube_pos=(0.5, 0.0, 0.1))
        _action, phase, _done = teacher.step(obs)
        assert phase == PHASE_MOVE_PREGRASP

    def test_move_pregrasp_to_visual_align(self):
        """Approaching within approach_align_threshold triggers VISUAL_ALIGN."""
        cfg = TeacherConfig(approach_align_threshold_m=0.15, pregrasp_height_offset=0.05)
        teacher = PickTeacher(config=cfg)

        # EE is close to pre-grasp position (cube_z + offset)
        cube_pos = (0.5, 0.0, 0.1)
        pregrasp_target = (0.5, 0.0, 0.1 + cfg.pregrasp_height_offset)
        # Place EE within threshold of pre-grasp target
        ee_pos = (pregrasp_target[0], pregrasp_target[1], pregrasp_target[2] + 0.05)
        obs = _make_obs(ee_pos=ee_pos, cube_pos=cube_pos)
        _action, phase, _done = teacher.step(obs)
        assert phase == PHASE_VISUAL_ALIGN

    def test_visual_align_to_execute_approach(self):
        """Close enough triggers EXECUTE_APPROACH."""
        cfg = TeacherConfig(execute_approach_threshold_m=0.05)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_VISUAL_ALIGN

        # EE very close to cube
        obs = _make_obs(ee_pos=(0.5, 0.0, 0.13), cube_pos=(0.5, 0.0, 0.1))
        _action, phase, _done = teacher.step(obs)
        assert phase == PHASE_EXECUTE_APPROACH

    def test_execute_approach_to_close_gripper(self):
        """Within grasp distance triggers CLOSE_GRIPPER."""
        cfg = TeacherConfig(grasp_distance_threshold_m=0.01)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_EXECUTE_APPROACH

        obs = _make_obs(ee_pos=(0.5, 0.0, 0.105), cube_pos=(0.5, 0.0, 0.1))
        _action, phase, _done = teacher.step(obs)
        assert phase == PHASE_CLOSE_GRIPPER

    def test_close_gripper_to_verify(self):
        """CLOSE_GRIPPER lasts close_gripper_steps then transitions."""
        cfg = TeacherConfig(close_gripper_steps=3, verify_steps=2, lift_steps=2)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_CLOSE_GRIPPER

        obs = _make_obs()
        for _ in range(2):
            _action, phase, _done = teacher.step(obs)
            assert phase == PHASE_CLOSE_GRIPPER
        _action, phase, _done = teacher.step(obs)
        assert phase == PHASE_VERIFY_GRASP

    def test_verify_to_lift(self):
        """VERIFY_GRASP lasts verify_steps then transitions."""
        cfg = TeacherConfig(verify_steps=2, lift_steps=5)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_VERIFY_GRASP

        obs = _make_obs()
        teacher.step(obs)
        _action, phase, _done = teacher.step(obs)
        assert phase == PHASE_LIFT

    def test_lift_to_done(self):
        """LIFT lasts lift_steps then transitions to DONE."""
        cfg = TeacherConfig(lift_steps=3)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_LIFT

        obs = _make_obs()
        for _ in range(2):
            _action, _phase, done = teacher.step(obs)
            assert not done
        _action, phase, done = teacher.step(obs)
        assert phase == PHASE_DONE
        assert done is True


# ---------------------------------------------------------------------------
# Action output
# ---------------------------------------------------------------------------


class TestActionOutput:
    def test_action_shape(self):
        teacher = PickTeacher()
        obs = _make_obs()
        action, _phase, _done = teacher.step(obs)
        assert action.shape == (ACTION_DIM,)

    def test_approach_action_moves_toward_target(self):
        """During MOVE_PREGRASP, action should point EE toward cube."""
        teacher = PickTeacher()
        # Cube below and ahead of EE
        obs = _make_obs(ee_pos=(0.5, 0.0, 0.5), cube_pos=(0.5, 0.0, 0.1))
        action, _phase, _done = teacher.step(obs)
        # z component should be negative (moving down)
        assert action[2] < 0

    def test_gripper_open_during_approach(self):
        """Gripper should be open during approach phases."""
        teacher = PickTeacher()
        obs = _make_obs(ee_pos=(0.5, 0.0, 0.5), cube_pos=(0.5, 0.0, 0.1))
        action, _phase, _done = teacher.step(obs)
        assert action[6] == GRIPPER_OPEN

    def test_gripper_closed_during_close_and_lift(self):
        """Gripper should be closed during CLOSE_GRIPPER and LIFT."""
        teacher = PickTeacher()
        teacher._phase = PHASE_CLOSE_GRIPPER

        obs = _make_obs()
        action, _phase, _done = teacher.step(obs)
        assert action[6] == GRIPPER_CLOSE

        teacher._phase = PHASE_LIFT
        teacher._phase_step = 0
        action, _phase, _done = teacher.step(obs)
        assert action[6] == GRIPPER_CLOSE

    def test_lift_action_moves_up(self):
        """During LIFT, action should have positive z."""
        teacher = PickTeacher()
        teacher._phase = PHASE_LIFT
        teacher._phase_step = 0

        obs = _make_obs()
        action, _phase, _done = teacher.step(obs)
        assert action[2] > 0

    def test_actions_clamped(self):
        """Actions should not exceed max_delta."""
        cfg = TeacherConfig(max_delta=0.02)
        teacher = PickTeacher(config=cfg)
        # Large distance = large raw delta
        obs = _make_obs(ee_pos=(1.0, 1.0, 1.0), cube_pos=(0.0, 0.0, 0.0))
        action, _phase, _done = teacher.step(obs)
        assert np.all(np.abs(action[:3]) <= cfg.max_delta + 1e-9)

    def test_zero_rotation_actions(self):
        """Teacher produces zero rotation deltas (position-only control)."""
        teacher = PickTeacher()
        obs = _make_obs(ee_pos=(0.5, 0.0, 0.5), cube_pos=(0.5, 0.0, 0.1))
        action, _phase, _done = teacher.step(obs)
        np.testing.assert_array_equal(action[3:6], 0.0)


# ---------------------------------------------------------------------------
# Full episode simulation
# ---------------------------------------------------------------------------


class TestFullEpisode:
    def test_completes_when_ee_reaches_cube(self):
        """Teacher should reach DONE if EE gradually approaches the cube."""
        cfg = TeacherConfig(
            close_gripper_steps=2,
            verify_steps=1,
            lift_steps=2,
        )
        teacher = PickTeacher(config=cfg)

        # Simulate convergence: EE approaches cube over steps
        positions = [
            # (ee_pos, cube_pos)
            ((0.5, 0.0, 0.5), (0.5, 0.0, 0.1)),  # far away → MOVE_PREGRASP
            ((0.5, 0.0, 0.19), (0.5, 0.0, 0.1)),  # within pregrasp → might still be MOVE_PREGRASP
            ((0.5, 0.0, 0.14), (0.5, 0.0, 0.1)),  # near → VISUAL_ALIGN
            ((0.5, 0.0, 0.13), (0.5, 0.0, 0.1)),  # closer → EXECUTE_APPROACH
            ((0.5, 0.0, 0.105), (0.5, 0.0, 0.1)),  # within grasp → CLOSE_GRIPPER
        ]

        phases_seen = []
        for ee_pos, cube_pos in positions:
            obs = _make_obs(ee_pos=ee_pos, cube_pos=cube_pos)
            _action, phase, done = teacher.step(obs)
            phases_seen.append(phase)
            if done:
                break

        # Should have progressed through approach phases
        assert PHASE_MOVE_PREGRASP in phases_seen

        # Now run timed phases if not done
        if not done:
            obs = _make_obs(ee_pos=(0.5, 0.0, 0.1), cube_pos=(0.5, 0.0, 0.1))
            for _ in range(100):
                _action, phase, done = teacher.step(obs)
                phases_seen.append(phase)
                if done:
                    break

        assert done, f"Teacher did not complete. Final phase: {teacher.phase}, seen: {phases_seen}"
        assert PHASE_DONE in phases_seen

    def test_phase_sequence_is_monotonic(self):
        """Phases should progress forward (no going back, since teacher has no recovery)."""
        cfg = TeacherConfig(
            close_gripper_steps=2,
            verify_steps=1,
            lift_steps=2,
        )
        teacher = PickTeacher(config=cfg)

        # Walk EE from far to close
        steps = [
            (0.5, 0.0, 0.5),
            (0.5, 0.0, 0.3),
            (0.5, 0.0, 0.2),
            (0.5, 0.0, 0.14),
            (0.5, 0.0, 0.13),
            (0.5, 0.0, 0.105),
        ]
        cube = (0.5, 0.0, 0.1)

        prev_phase = -1
        for ee_pos in steps:
            obs = _make_obs(ee_pos=ee_pos, cube_pos=cube)
            _action, phase, _done = teacher.step(obs)
            assert phase >= prev_phase, f"Phase went backward: {prev_phase} → {phase}"
            prev_phase = phase
