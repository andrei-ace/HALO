"""Tests for PickTeacher scripted policy.

Uses a real MuJoCo model (SO-101 pick_scene.xml) for IK-dependent tests.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

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
# Fixtures
# ---------------------------------------------------------------------------

_SCENE_XML = str(Path(__file__).resolve().parent.parent / "assets" / "so101" / "pick_scene.xml")


@pytest.fixture(scope="module")
def mj_model():
    """Load the SO-101 pick scene model once per module."""
    return mujoco.MjModel.from_xml_path(_SCENE_XML)


@pytest.fixture()
def mj_data(mj_model):
    """Fresh MuJoCo data for each test."""
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(
    mj_model,
    mj_data,
    *,
    ee_pos: tuple[float, float, float] | None = None,
    cube_pos: tuple[float, float, float] = (0.2, 0.0, 0.025),
    gripper: float = 0.0,
) -> dict:
    """Create a minimal observation dict for teacher testing.

    If ee_pos is None, uses the current EE position from mj_data.
    """
    if ee_pos is None:
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        ee_pos_arr = mj_data.site_xpos[site_id].copy()
    else:
        ee_pos_arr = np.array(ee_pos)

    ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
    cube_quat = np.array([1.0, 0.0, 0.0, 0.0])
    return {
        "ee_pose": np.concatenate([ee_pos_arr, ee_quat]),
        "object_pose": np.concatenate([np.array(cube_pos), cube_quat]),
        "gripper": gripper,
        "joint_pos": mj_data.qpos[:6].copy(),
    }


# ---------------------------------------------------------------------------
# Init and reset
# ---------------------------------------------------------------------------


class TestPickTeacherInit:
    def test_initial_phase_is_idle(self):
        teacher = PickTeacher()
        assert teacher.phase == PHASE_IDLE
        assert teacher.done is False

    def test_reset_returns_to_idle(self, mj_model, mj_data):
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data, cube_pos=(0.2, 0.0, 0.025))
        teacher.step(obs, mj_model, mj_data)  # advance past IDLE
        assert teacher.phase != PHASE_IDLE
        teacher.reset()
        assert teacher.phase == PHASE_IDLE

    def test_reset_clears_grasp_target(self):
        teacher = PickTeacher()
        teacher._grasp_target = np.array([1.0, 2.0, 3.0])
        teacher._pregrasp_wrist_joints = np.array([0.1, 0.2])
        teacher.reset()
        assert teacher._grasp_target is None
        assert teacher._pregrasp_wrist_joints is None

    def test_custom_config(self):
        cfg = TeacherConfig(approach_align_threshold_m=0.20)
        teacher = PickTeacher(config=cfg)
        assert teacher._config.approach_align_threshold_m == 0.20


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------


class TestPhaseTransitions:
    def test_idle_to_move_pregrasp_immediate(self, mj_model, mj_data):
        """IDLE → SELECT_GRASP → PLAN_APPROACH → MOVE_PREGRASP on first step."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data, ee_pos=(0.39, 0.0, 0.23), cube_pos=(0.2, 0.0, 0.025))
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_MOVE_PREGRASP

    def test_move_pregrasp_to_visual_align(self, mj_model, mj_data):
        """Approaching within approach_align_threshold triggers VISUAL_ALIGN."""
        cfg = TeacherConfig(approach_align_threshold_m=0.03, pregrasp_height_offset=0.05)
        teacher = PickTeacher(config=cfg)

        # EE close to pre-grasp position (cube_z + offset = 0.075)
        cube_pos = (0.2, 0.0, 0.025)
        ee_pos = (0.2, 0.0, 0.08)  # within 0.03 of pregrasp target
        obs = _make_obs(mj_model, mj_data, ee_pos=ee_pos, cube_pos=cube_pos)
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_VISUAL_ALIGN

    def test_visual_align_to_execute_approach(self, mj_model, mj_data):
        """XY-aligned above cube triggers EXECUTE_APPROACH."""
        cfg = TeacherConfig(execute_approach_threshold_m=0.05, pregrasp_height_offset=0.06)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_VISUAL_ALIGN

        # EE above cube — XY aligned, Z at pregrasp height (doesn't matter for XY check)
        cube_pos = (0.2, 0.0, 0.025)
        ee_pos = (0.2, 0.0, 0.085)  # directly above, XY dist = 0
        obs = _make_obs(mj_model, mj_data, ee_pos=ee_pos, cube_pos=cube_pos)
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_EXECUTE_APPROACH

    def test_execute_approach_to_close_gripper(self, mj_model, mj_data):
        """Within grasp distance of target triggers CLOSE_GRIPPER."""
        cfg = TeacherConfig(grasp_distance_threshold_m=0.025, grasp_height_offset=-0.005)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_EXECUTE_APPROACH

        # Target = cube_z + offset = 0.025 - 0.005 = 0.020; EE at 0.02 → dist ~= 0
        obs = _make_obs(mj_model, mj_data, ee_pos=(0.2, 0.0, 0.02), cube_pos=(0.2, 0.0, 0.025))
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_CLOSE_GRIPPER

    def test_visual_align_stores_pregrasp_wrist_joints(self, mj_model, mj_data):
        """Transitioning to EXECUTE_APPROACH stores the wrist joint angles for orientation lock."""
        cfg = TeacherConfig(execute_approach_threshold_m=0.05, pregrasp_height_offset=0.06)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_VISUAL_ALIGN

        cube_pos = (0.2, 0.0, 0.025)
        ee_pos = (0.2, 0.0, 0.085)  # XY-aligned
        obs = _make_obs(mj_model, mj_data, ee_pos=ee_pos, cube_pos=cube_pos)
        teacher.step(obs, mj_model, mj_data)
        assert teacher.phase == PHASE_EXECUTE_APPROACH
        assert teacher._pregrasp_wrist_joints is not None
        assert teacher._pregrasp_wrist_joints.shape == (2,)

    def test_execute_approach_stores_grasp_target(self, mj_model, mj_data):
        """Transitioning to CLOSE_GRIPPER stores the grasp target for IK hold."""
        cfg = TeacherConfig(grasp_distance_threshold_m=0.025, grasp_height_offset=-0.005)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_EXECUTE_APPROACH

        cube_pos = (0.2, 0.0, 0.025)
        ee_pos = (0.2, 0.0, 0.02)
        obs = _make_obs(mj_model, mj_data, ee_pos=ee_pos, cube_pos=cube_pos)
        teacher.step(obs, mj_model, mj_data)

        assert teacher._grasp_target is not None
        expected = np.array([0.2, 0.0, 0.025 - 0.005])
        np.testing.assert_allclose(teacher._grasp_target, expected)

    def test_close_gripper_to_verify(self, mj_model, mj_data):
        """CLOSE_GRIPPER lasts close_gripper_steps then transitions."""
        cfg = TeacherConfig(close_gripper_steps=3, verify_steps=2, lift_steps=2)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_CLOSE_GRIPPER

        obs = _make_obs(mj_model, mj_data)
        for _ in range(2):
            _action, phase, _done = teacher.step(obs, mj_model, mj_data)
            assert phase == PHASE_CLOSE_GRIPPER
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_VERIFY_GRASP

    def test_verify_to_lift(self, mj_model, mj_data):
        """VERIFY_GRASP lasts verify_steps then transitions."""
        cfg = TeacherConfig(verify_steps=2, lift_steps=5)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_VERIFY_GRASP

        obs = _make_obs(mj_model, mj_data)
        teacher.step(obs, mj_model, mj_data)
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_LIFT

    def test_lift_to_done(self, mj_model, mj_data):
        """LIFT lasts lift_steps then transitions to DONE."""
        cfg = TeacherConfig(lift_steps=3)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_LIFT

        obs = _make_obs(mj_model, mj_data)
        for _ in range(2):
            _action, _phase, done = teacher.step(obs, mj_model, mj_data)
            assert not done
        _action, phase, done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_DONE
        assert done is True


# ---------------------------------------------------------------------------
# Speed clamping
# ---------------------------------------------------------------------------


class TestSpeedClamping:
    def test_joint_delta_clamped(self, mj_model, mj_data):
        """Arm joints should not jump more than max_joint_delta per step."""
        cfg = TeacherConfig(max_joint_delta=0.05)
        teacher = PickTeacher(config=cfg)

        # EE far from cube — IK wants a big jump
        obs = _make_obs(mj_model, mj_data, ee_pos=(0.39, 0.0, 0.23), cube_pos=(0.2, 0.0, 0.025))
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)

        current = obs["joint_pos"][:5]
        delta = np.abs(action[:5] - current)
        np.testing.assert_array_less(delta, cfg.max_joint_delta + 1e-9)

    def test_no_clamping_when_close(self, mj_model, mj_data):
        """When IK target is very close to current, clamping has no effect."""
        cfg = TeacherConfig(max_joint_delta=0.05)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_EXECUTE_APPROACH

        # EE at cube — IK delta is near zero
        obs = _make_obs(mj_model, mj_data, ee_pos=(0.2, 0.0, 0.025), cube_pos=(0.2, 0.0, 0.025))
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)
        assert action.shape == (ACTION_DIM,)

    def test_large_max_delta_allows_full_jump(self, mj_model, mj_data):
        """With a large max_joint_delta, the output matches raw IK."""
        cfg = TeacherConfig(max_joint_delta=100.0)  # effectively unclamped
        teacher = PickTeacher(config=cfg)

        obs = _make_obs(mj_model, mj_data, ee_pos=(0.39, 0.0, 0.23), cube_pos=(0.2, 0.0, 0.025))
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)

        # With no clamping, delta can be large
        current = obs["joint_pos"][:5]
        delta = np.abs(action[:5] - current)
        assert np.max(delta) > 0.05  # at least some joint moved significantly

    def test_gripper_delta_clamped(self, mj_model, mj_data):
        """Gripper should not jump more than max_gripper_delta per step."""
        cfg = TeacherConfig(max_gripper_delta=0.04)
        teacher = PickTeacher()
        teacher._config = cfg
        teacher._phase = PHASE_CLOSE_GRIPPER

        # Start with gripper fully open
        obs = _make_obs(mj_model, mj_data, gripper=GRIPPER_OPEN)
        obs["joint_pos"] = mj_data.qpos[:6].copy()
        obs["joint_pos"][5] = GRIPPER_OPEN
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)

        gripper_delta = abs(action[5] - GRIPPER_OPEN)
        assert gripper_delta <= cfg.max_gripper_delta + 1e-9


# ---------------------------------------------------------------------------
# Action output
# ---------------------------------------------------------------------------


class TestActionOutput:
    def test_action_shape(self, mj_model, mj_data):
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)
        assert action.shape == (ACTION_DIM,)

    def test_gripper_moves_toward_open_during_approach(self, mj_model, mj_data):
        """Gripper should move toward GRIPPER_OPEN during approach phases."""
        for phase in (PHASE_MOVE_PREGRASP, PHASE_VISUAL_ALIGN, PHASE_EXECUTE_APPROACH):
            teacher = PickTeacher()
            teacher._phase = phase
            # Set gripper joint to a value below GRIPPER_OPEN so clamping moves toward open
            mj_data_copy = mujoco.MjData(mj_model)
            mj_data_copy.qpos[:] = mj_data.qpos[:]
            mj_data_copy.qpos[5] = 0.5  # midway — should move toward GRIPPER_OPEN (1.75)
            mujoco.mj_forward(mj_model, mj_data_copy)
            obs = _make_obs(mj_model, mj_data_copy, ee_pos=(0.2, 0.0, 0.05), cube_pos=(0.2, 0.0, 0.025))
            obs["joint_pos"] = mj_data_copy.qpos[:6].copy()
            action, _phase, _done = teacher.step(obs, mj_model, mj_data_copy)
            # Gripper should be moving toward GRIPPER_OPEN (increasing)
            assert action[5] > 0.5, f"Gripper not moving toward open in phase {phase}"
            assert action[5] <= GRIPPER_OPEN + 1e-6, f"Gripper exceeded open in phase {phase}"

    def test_gripper_moves_toward_close_during_close_phase(self, mj_model, mj_data):
        """Gripper should gradually close (not instantly) during CLOSE_GRIPPER."""
        teacher = PickTeacher()
        teacher._phase = PHASE_CLOSE_GRIPPER

        # Start with gripper at GRIPPER_OPEN
        mj_data_copy = mujoco.MjData(mj_model)
        mj_data_copy.qpos[:] = mj_data.qpos[:]
        mj_data_copy.qpos[5] = GRIPPER_OPEN
        mujoco.mj_forward(mj_model, mj_data_copy)
        obs = _make_obs(mj_model, mj_data_copy)
        obs["joint_pos"] = mj_data_copy.qpos[:6].copy()
        obs["joint_pos"][5] = GRIPPER_OPEN
        action, _phase, _done = teacher.step(obs, mj_model, mj_data_copy)

        # Should move toward GRIPPER_CLOSE but NOT jump there instantly
        assert action[5] < GRIPPER_OPEN, "Gripper should be moving toward closed"
        assert action[5] > GRIPPER_CLOSE, "Gripper should not jump to fully closed in one step"

    def test_close_gripper_solves_ik_to_grasp_target(self, mj_model, mj_data):
        """CLOSE_GRIPPER should actively hold position via IK and gradually close gripper."""
        cfg = TeacherConfig(close_gripper_steps=5, grasp_distance_threshold_m=0.025, grasp_height_offset=-0.005)
        teacher = PickTeacher(config=cfg)

        # Transition through EXECUTE_APPROACH to set _grasp_target
        teacher._phase = PHASE_EXECUTE_APPROACH
        cube_pos = (0.2, 0.0, 0.025)
        ee_pos = (0.2, 0.0, 0.02)
        obs = _make_obs(mj_model, mj_data, ee_pos=ee_pos, cube_pos=cube_pos)
        teacher.step(obs, mj_model, mj_data)
        assert teacher.phase == PHASE_CLOSE_GRIPPER
        assert teacher._grasp_target is not None

        # Step in CLOSE_GRIPPER — should produce IK-solved joints, gripper moving toward closed
        obs2 = _make_obs(mj_model, mj_data, ee_pos=(0.2, 0.0, 0.03), cube_pos=cube_pos)
        action, _phase, _done = teacher.step(obs2, mj_model, mj_data)
        assert action.shape == (ACTION_DIM,)
        # Gripper moving toward GRIPPER_CLOSE (may not be there yet due to clamping)
        assert action[5] <= GRIPPER_OPEN


# ---------------------------------------------------------------------------
# Full episode simulation
# ---------------------------------------------------------------------------


class TestFullEpisode:
    def test_completes_when_ee_reaches_cube(self, mj_model, mj_data):
        """Teacher should reach DONE if EE gradually approaches the cube."""
        cfg = TeacherConfig(
            close_gripper_steps=2,
            verify_steps=1,
            lift_steps=2,
        )
        teacher = PickTeacher(config=cfg)

        # Simulate trajectory: far → above cube → settle → descend → grasp
        cube = (0.2, 0.0, 0.025)
        pregrasp_z = 0.025 + cfg.pregrasp_height_offset
        grasp_z = 0.025 + cfg.grasp_height_offset
        positions = [
            (0.39, 0.0, 0.23),  # far away → MOVE_PREGRASP
            (0.2, 0.0, pregrasp_z + 0.01),  # near pregrasp → VISUAL_ALIGN
            (0.2, 0.0, 0.05),  # near cube → EXECUTE_APPROACH
            (0.2, 0.0, grasp_z + 0.005),  # close to grasp target
            (0.2, 0.0, grasp_z),  # at grasp target → CLOSE_GRIPPER
        ]

        phases_seen = []
        for ee_pos in positions:
            obs = _make_obs(mj_model, mj_data, ee_pos=ee_pos, cube_pos=cube)
            _action, phase, done = teacher.step(obs, mj_model, mj_data)
            phases_seen.append(phase)
            if done:
                break

        # Should have progressed through approach phases
        assert PHASE_MOVE_PREGRASP in phases_seen

        # Now run timed phases if not done
        if not done:
            obs = _make_obs(mj_model, mj_data, ee_pos=(0.2, 0.0, grasp_z), cube_pos=cube)
            for _ in range(100):
                _action, phase, done = teacher.step(obs, mj_model, mj_data)
                phases_seen.append(phase)
                if done:
                    break

        assert done, f"Teacher did not complete. Final phase: {teacher.phase}, seen: {phases_seen}"
        assert PHASE_DONE in phases_seen

    def test_phase_sequence_is_monotonic(self, mj_model, mj_data):
        """Phases should progress forward (no going back, since teacher has no recovery)."""
        cfg = TeacherConfig(
            close_gripper_steps=2,
            verify_steps=1,
            lift_steps=2,
        )
        teacher = PickTeacher(config=cfg)

        cube = (0.2, 0.0, 0.025)
        pregrasp_z = 0.025 + cfg.pregrasp_height_offset
        grasp_z = 0.025 + cfg.grasp_height_offset

        # Walk EE: far → above cube → near → at grasp height
        steps = [
            (0.39, 0.0, 0.23),
            (0.25, 0.0, 0.15),
            (0.2, 0.0, pregrasp_z + 0.01),  # near pregrasp
            (0.2, 0.0, 0.05),
            (0.2, 0.0, grasp_z + 0.005),
            (0.2, 0.0, grasp_z),
        ]

        prev_phase = -1
        for ee_pos in steps:
            obs = _make_obs(mj_model, mj_data, ee_pos=ee_pos, cube_pos=cube)
            _action, phase, _done = teacher.step(obs, mj_model, mj_data)
            assert phase >= prev_phase, f"Phase went backward: {prev_phase} → {phase}"
            prev_phase = phase
