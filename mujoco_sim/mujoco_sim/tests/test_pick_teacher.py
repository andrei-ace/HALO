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
        cfg = TeacherConfig(approach_align_threshold_m=0.15, pregrasp_height_offset=0.05)
        teacher = PickTeacher(config=cfg)

        # EE close to pre-grasp position (cube_z + offset)
        cube_pos = (0.2, 0.0, 0.025)
        # Pre-grasp target is at (0.2, 0, 0.075); place EE within threshold
        ee_pos = (0.2, 0.0, 0.12)
        obs = _make_obs(mj_model, mj_data, ee_pos=ee_pos, cube_pos=cube_pos)
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_VISUAL_ALIGN

    def test_visual_align_to_execute_approach(self, mj_model, mj_data):
        """Close enough triggers EXECUTE_APPROACH."""
        cfg = TeacherConfig(execute_approach_threshold_m=0.05)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_VISUAL_ALIGN

        # EE very close to cube
        obs = _make_obs(mj_model, mj_data, ee_pos=(0.2, 0.0, 0.05), cube_pos=(0.2, 0.0, 0.025))
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_EXECUTE_APPROACH

    def test_execute_approach_to_close_gripper(self, mj_model, mj_data):
        """Within grasp distance of target triggers CLOSE_GRIPPER."""
        cfg = TeacherConfig(grasp_distance_threshold_m=0.01, grasp_height_offset=-0.01)
        teacher = PickTeacher(config=cfg)
        teacher._phase = PHASE_EXECUTE_APPROACH

        # Target = cube_z + offset = 0.025 - 0.01 = 0.015; EE at 0.02 → dist ~= 0.005
        obs = _make_obs(mj_model, mj_data, ee_pos=(0.2, 0.0, 0.02), cube_pos=(0.2, 0.0, 0.025))
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        assert phase == PHASE_CLOSE_GRIPPER

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
# Action output
# ---------------------------------------------------------------------------


class TestActionOutput:
    def test_action_shape(self, mj_model, mj_data):
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)
        assert action.shape == (ACTION_DIM,)

    def test_gripper_open_during_execute_approach(self, mj_model, mj_data):
        """Gripper should open during EXECUTE_APPROACH to straddle the cube."""
        teacher = PickTeacher()
        teacher._phase = PHASE_EXECUTE_APPROACH
        obs = _make_obs(mj_model, mj_data, ee_pos=(0.2, 0.0, 0.05), cube_pos=(0.2, 0.0, 0.025))
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)
        assert action[5] == GRIPPER_OPEN

    def test_gripper_closed_during_close_and_lift(self, mj_model, mj_data):
        """Gripper should be closed during CLOSE_GRIPPER and LIFT."""
        teacher = PickTeacher()
        teacher._phase = PHASE_CLOSE_GRIPPER

        obs = _make_obs(mj_model, mj_data)
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)
        assert action[5] == GRIPPER_CLOSE

        teacher._phase = PHASE_LIFT
        teacher._phase_step = 0
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)
        assert action[5] == GRIPPER_CLOSE


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

        # Simulate convergence: EE approaches cube over steps
        cube = (0.2, 0.0, 0.025)
        positions = [
            (0.39, 0.0, 0.23),  # far away → MOVE_PREGRASP
            (0.2, 0.0, 0.12),  # within pregrasp
            (0.2, 0.0, 0.05),  # near → VISUAL_ALIGN / EXECUTE_APPROACH
            (0.2, 0.0, 0.03),  # within grasp distance
            (0.2, 0.0, 0.02),  # close → CLOSE_GRIPPER
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
            obs = _make_obs(mj_model, mj_data, ee_pos=(0.2, 0.0, 0.025), cube_pos=cube)
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

        # Walk EE from far to close
        cube = (0.2, 0.0, 0.025)
        steps = [
            (0.39, 0.0, 0.23),
            (0.25, 0.0, 0.15),
            (0.2, 0.0, 0.10),
            (0.2, 0.0, 0.05),
            (0.2, 0.0, 0.03),
            (0.2, 0.0, 0.02),
        ]

        prev_phase = -1
        for ee_pos in steps:
            obs = _make_obs(mj_model, mj_data, ee_pos=ee_pos, cube_pos=cube)
            _action, phase, _done = teacher.step(obs, mj_model, mj_data)
            assert phase >= prev_phase, f"Phase went backward: {prev_phase} → {phase}"
            prev_phase = phase
