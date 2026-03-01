"""Tests for PickTeacher trajectory-planned policy.

Uses a real MuJoCo model (SO-101 pick_scene.xml) for IK-dependent tests.
The teacher now pre-computes a full trajectory on the first step() call,
then samples it at control rate.
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
)
from mujoco_sim.scene_info import SceneInfo
from mujoco_sim.teacher.pick_teacher import PickTeacher, TeacherConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCENE_XML = str(Path(__file__).resolve().parent.parent / "assets" / "so101" / "pick_scene.xml")


@pytest.fixture(scope="module")
def mj_model():
    """Load the SO-101 pick scene model once per module."""
    return mujoco.MjModel.from_xml_path(_SCENE_XML)


@pytest.fixture(scope="module")
def scene_info(mj_model):
    return SceneInfo.from_model(mj_model)


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
    cube_pos: tuple[float, float, float] | None = None,
) -> dict:
    """Create an observation dict from current MuJoCo state.

    If cube_pos is None, reads the cube position from MuJoCo data (default).
    """
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    ee_pos = mj_data.site_xpos[site_id].copy()
    ee_xmat = mj_data.site_xmat[site_id].reshape(3, 3)
    ee_quat = np.zeros(4)
    mujoco.mju_mat2Quat(ee_quat, ee_xmat.flatten())
    ee_pose = np.concatenate([ee_pos, ee_quat])

    if cube_pos is None:
        cube_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_pos_arr = mj_data.xpos[cube_body].copy()
        cube_quat = mj_data.xquat[cube_body].copy()
    else:
        cube_pos_arr = np.array(cube_pos)
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

    return {
        "ee_pose": ee_pose,
        "object_pose": np.concatenate([cube_pos_arr, cube_quat]),
        "gripper": float(mj_data.qpos[5]),
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
        obs = _make_obs(mj_model, mj_data)
        teacher.step(obs, mj_model, mj_data)
        teacher.reset()
        assert teacher.phase == PHASE_IDLE
        assert teacher._plan is None

    def test_custom_config(self):
        cfg = TeacherConfig(pregrasp_height_offset=0.20)
        teacher = PickTeacher(config=cfg)
        assert teacher._config.pregrasp_height_offset == 0.20


# ---------------------------------------------------------------------------
# Phase transitions (trajectory-based)
# ---------------------------------------------------------------------------


class TestPhaseTransitions:
    def test_first_step_plans_and_returns_phase(self, mj_model, mj_data):
        """First step should plan trajectory and return a valid phase."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)
        _action, phase, _done = teacher.step(obs, mj_model, mj_data)
        # First step samples at t=0, which is the first segment
        assert phase in (PHASE_IDLE, PHASE_MOVE_PREGRASP)
        assert teacher._plan is not None

    def test_phases_are_monotonic(self, mj_model, mj_data):
        """Phases should only progress forward over the trajectory."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)

        prev_phase = -1
        for _ in range(2000):
            _action, phase, done = teacher.step(obs, mj_model, mj_data)
            assert phase >= prev_phase, f"Phase went backward: {prev_phase} → {phase}"
            prev_phase = phase
            if done:
                break

        assert done, f"Teacher did not complete. Final phase: {teacher.phase}"

    def test_expected_phase_sequence(self, mj_model, mj_data):
        """Trajectory should produce the expected phase sequence."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)

        phases_seen = set()
        for _ in range(2000):
            _action, phase, done = teacher.step(obs, mj_model, mj_data)
            phases_seen.add(phase)
            if done:
                break

        # Must see all main phases (DONE is set on completion)
        assert PHASE_MOVE_PREGRASP in phases_seen
        assert PHASE_EXECUTE_APPROACH in phases_seen
        assert PHASE_CLOSE_GRIPPER in phases_seen
        assert PHASE_LIFT in phases_seen
        assert PHASE_DONE in phases_seen


# ---------------------------------------------------------------------------
# Action output
# ---------------------------------------------------------------------------


class TestActionOutput:
    def test_action_shape(self, mj_model, mj_data):
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)
        assert action.shape == (ACTION_DIM,)

    def test_actions_are_continuous(self, mj_model, mj_data):
        """Consecutive actions should be close (no sudden jumps)."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)

        prev_action = None
        max_delta = 0.0
        for _ in range(2000):
            action, _phase, done = teacher.step(obs, mj_model, mj_data)
            if prev_action is not None:
                delta = np.max(np.abs(action[:5] - prev_action[:5]))
                max_delta = max(max_delta, delta)
            prev_action = action.copy()
            if done:
                break

        # At 20 Hz with max_vel=1.5 rad/s, max step delta = 1.5 * 0.05 = 0.075 rad
        # Allow some margin for ruckig's jerk profile
        assert max_delta < 0.15, f"Max arm joint delta per step: {max_delta:.4f}"

    def test_gripper_starts_open(self, mj_model, mj_data):
        """First action should have gripper near open."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)
        action, _phase, _done = teacher.step(obs, mj_model, mj_data)
        # Gripper should be near GRIPPER_OPEN at the start
        assert abs(action[5] - GRIPPER_OPEN) < 0.5

    def test_gripper_closes_during_trajectory(self, mj_model, mj_data):
        """Gripper should transition from open to close during the trajectory."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)

        saw_open = False
        saw_closed = False
        for _ in range(2000):
            action, _phase, done = teacher.step(obs, mj_model, mj_data)
            if action[5] > GRIPPER_OPEN - 0.5:
                saw_open = True
            if action[5] < GRIPPER_CLOSE + 0.5:
                saw_closed = True
            if done:
                break

        assert saw_open, "Never saw gripper near open"
        assert saw_closed, "Never saw gripper near closed"


# ---------------------------------------------------------------------------
# Full episode simulation
# ---------------------------------------------------------------------------


class TestFullEpisode:
    def test_completes_within_budget(self, mj_model, mj_data):
        """Teacher should reach DONE within a reasonable step budget."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)

        for step in range(2000):
            _action, _phase, done = teacher.step(obs, mj_model, mj_data)
            if done:
                break

        assert done, f"Teacher did not complete after {step + 1} steps. Phase: {teacher.phase}"
        # Trajectory should complete in a reasonable number of steps
        assert step < 1500, f"Took too many steps: {step + 1}"

    def test_plan_duration_reasonable(self, mj_model, mj_data):
        """Trajectory total duration should be reasonable (1-30 seconds)."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)
        teacher.step(obs, mj_model, mj_data)
        assert teacher._plan is not None
        assert 0.5 < teacher._plan.total_duration < 30.0

    def test_different_cube_positions(self, scene_info, mj_model, mj_data):
        """Teacher should complete for various cube placements on the table."""
        cx, cy, cube_z = scene_info.cube_default_pos
        for cube_pos in [
            (float(cx), float(cy), float(cube_z)),
            (float(cx + 0.01), float(cy + 0.01), float(cube_z)),
            (float(cx - 0.01), float(cy - 0.01), float(cube_z)),
        ]:
            teacher = PickTeacher()
            obs = _make_obs(mj_model, mj_data, cube_pos=cube_pos)
            for _ in range(2000):
                _action, _phase, done = teacher.step(obs, mj_model, mj_data)
                if done:
                    break
            assert done, f"Teacher failed for cube_pos={cube_pos}"

    def test_reset_allows_reuse(self, mj_model, mj_data):
        """After reset, teacher should plan a fresh trajectory."""
        teacher = PickTeacher()
        obs = _make_obs(mj_model, mj_data)

        # First episode
        teacher.step(obs, mj_model, mj_data)

        # Reset and run again
        teacher.reset()
        assert teacher._plan is None

        teacher.step(obs, mj_model, mj_data)
        assert teacher._plan is not None
        # Should produce a valid freshly computed plan (duration may differ due to random sampling)
        assert 0.5 < teacher._plan.total_duration < 30.0
