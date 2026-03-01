"""Tests for the trajectory-planned pick pipeline (keyframes → IK → ruckig).

Uses a real MuJoCo model (SO-101 pick_scene.xml) for IK-dependent tests.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_sim.constants import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    PHASE_CLOSE_GRIPPER,
    PHASE_EXECUTE_APPROACH,
    PHASE_IDLE,
    PHASE_LIFT,
    PHASE_MOVE_PREGRASP,
)
from mujoco_sim.teacher.keyframe_planner import _TCP_PINCH_OFFSET_LOCAL, plan_pick_keyframes
from mujoco_sim.teacher.trajectory import JointLimits, plan_trajectory
from mujoco_sim.teacher.waypoint_generator import JointWaypoint, generate_joint_waypoints

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCENE_XML = str(Path(__file__).resolve().parent.parent / "assets" / "so101" / "pick_scene.xml")


@pytest.fixture(scope="module")
def mj_model():
    return mujoco.MjModel.from_xml_path(_SCENE_XML)


@pytest.fixture()
def mj_data(mj_model):
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)
    return data


@pytest.fixture()
def ee_site_id(mj_model):
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")


@pytest.fixture()
def arm_joint_ids(mj_model):
    return [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    ]


# ---------------------------------------------------------------------------
# Keyframe planner tests
# ---------------------------------------------------------------------------


class TestKeyframePlanner:
    def test_produces_five_keyframes(self, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)
        assert len(keyframes) == 5

    def test_keyframe_labels(self, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)
        labels = [kf.label for kf in keyframes]
        assert labels == ["home", "pregrasp", "grasp", "grasp_closed", "lift"]

    def test_keyframe_phases(self, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)
        phases = [kf.phase_id for kf in keyframes]
        assert phases == [PHASE_IDLE, PHASE_MOVE_PREGRASP, PHASE_EXECUTE_APPROACH, PHASE_CLOSE_GRIPPER, PHASE_LIFT]

    def test_pregrasp_above_cube(self, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])
        z_hover = 0.15

        keyframes = plan_pick_keyframes(
            home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data, z_hover=z_hover
        )

        pregrasp = keyframes[1]
        # Gripperframe is offset from cube so that jaw midpoint lands at contact point
        grasp_rot = pregrasp.orientation
        offset_world = grasp_rot @ _TCP_PINCH_OFFSET_LOCAL
        expected_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + z_hover]) - offset_world
        np.testing.assert_allclose(pregrasp.position, expected_pos)

    def test_grasp_at_cube(self, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)

        grasp = keyframes[2]
        # Gripperframe is offset from cube so that jaw midpoint lands at cube center
        grasp_rot = grasp.orientation
        offset_world = grasp_rot @ _TCP_PINCH_OFFSET_LOCAL
        expected_pos = cube_pos - offset_world
        np.testing.assert_allclose(grasp.position, expected_pos)

    def test_gripper_sequence(self, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)

        # open, open, open, close, close
        expected_grippers = [GRIPPER_OPEN, GRIPPER_OPEN, GRIPPER_OPEN, GRIPPER_CLOSE, GRIPPER_CLOSE]
        actual = [kf.gripper for kf in keyframes]
        assert actual == expected_grippers

    def test_orientation_z_axis_points_down(self, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)

        # All non-home keyframes should have Z-axis pointing down
        for kf in keyframes[1:]:
            z_axis = kf.orientation[:, 2]
            np.testing.assert_allclose(z_axis, [0.0, 0.0, -1.0], atol=1e-10)

    def test_tcp_offset_applied(self, mj_model, mj_data, ee_site_id):
        """Grasp keyframe position differs from cube_pos by the TCP pinch offset."""
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)

        grasp = keyframes[2]
        # Reconstruct: grasp.position + grasp_rot @ offset should equal cube_pos
        grasp_rot = grasp.orientation
        jaw_midpoint = grasp.position + grasp_rot @ _TCP_PINCH_OFFSET_LOCAL
        np.testing.assert_allclose(jaw_midpoint, cube_pos, atol=1e-10)
        # The offset should be non-zero
        diff = grasp.position - cube_pos
        assert np.linalg.norm(diff) > 1e-6, "Grasp keyframe should be offset from cube center"

    def test_orientation_varies_with_cube_yaw(self, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])

        # Identity quaternion (yaw=0)
        kf0 = plan_pick_keyframes(home_joints, cube_pos, np.array([1.0, 0.0, 0.0, 0.0]), ee_site_id, mj_model, mj_data)
        # 45-degree yaw quaternion (w=cos(22.5°), z=sin(22.5°))
        angle = np.pi / 4
        quat45 = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        kf45 = plan_pick_keyframes(home_joints, cube_pos, quat45, ee_site_id, mj_model, mj_data)

        # X-axis should differ between the two
        x0 = kf0[1].orientation[:, 0]
        x45 = kf45[1].orientation[:, 0]
        assert not np.allclose(x0, x45, atol=1e-3)


# ---------------------------------------------------------------------------
# Waypoint generator tests
# ---------------------------------------------------------------------------


class TestWaypointGenerator:
    def test_produces_correct_count(self, mj_model, mj_data, ee_site_id, arm_joint_ids):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
        )

        assert len(waypoints) == len(keyframes)

    def test_waypoint_labels_match_keyframes(self, mj_model, mj_data, ee_site_id, arm_joint_ids):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
        )

        for kf, wp in zip(keyframes, waypoints):
            assert kf.label == wp.label
            assert kf.phase_id == wp.phase_id
            assert kf.gripper == wp.gripper

    def test_arm_joints_shape(self, mj_model, mj_data, ee_site_id, arm_joint_ids):
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
        )

        for wp in waypoints:
            assert wp.arm_joints.shape == (5,)

    def test_ik_reaches_keyframe_positions(self, mj_model, mj_data, ee_site_id, arm_joint_ids):
        """Verify FK of IK solution is close to keyframe target."""
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
        )

        d_check = mujoco.MjData(mj_model)
        for kf, wp in zip(keyframes, waypoints):
            d_check.qpos[:] = mj_data.qpos[:]
            for i, jid in enumerate(arm_joint_ids):
                d_check.qpos[jid] = wp.arm_joints[i]
            mujoco.mj_forward(mj_model, d_check)
            ee_pos = d_check.site_xpos[ee_site_id]
            err = float(np.linalg.norm(kf.position - ee_pos))
            assert err < 0.01, f"IK error for '{kf.label}': {err:.4f} m"


# ---------------------------------------------------------------------------
# Trajectory planner tests
# ---------------------------------------------------------------------------


class TestTrajectoryPlanner:
    def _make_waypoints(self) -> list[JointWaypoint]:
        """Create simple test waypoints."""
        q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.5, -0.3, 0.2, 0.1, 0.0])
        q2 = np.array([0.5, -0.5, 0.4, 0.2, 0.0])
        return [
            JointWaypoint(q0, GRIPPER_OPEN, PHASE_IDLE, "start"),
            JointWaypoint(q1, GRIPPER_OPEN, PHASE_MOVE_PREGRASP, "pregrasp"),
            JointWaypoint(q2, GRIPPER_OPEN, PHASE_EXECUTE_APPROACH, "grasp"),
            JointWaypoint(q2.copy(), GRIPPER_CLOSE, PHASE_CLOSE_GRIPPER, "grasp_closed"),
            JointWaypoint(q1.copy(), GRIPPER_CLOSE, PHASE_LIFT, "lift"),
        ]

    def test_plan_produces_segments(self):
        waypoints = self._make_waypoints()
        plan = plan_trajectory(waypoints)
        assert len(plan.segments) == len(waypoints) - 1

    def test_total_duration_positive(self):
        waypoints = self._make_waypoints()
        plan = plan_trajectory(waypoints)
        assert plan.total_duration > 0

    def test_sample_at_zero(self):
        waypoints = self._make_waypoints()
        plan = plan_trajectory(waypoints)
        arm, gripper, phase = plan.sample(0.0)
        assert arm.shape == (5,)
        np.testing.assert_allclose(arm, waypoints[0].arm_joints, atol=1e-4)

    def test_sample_at_end(self):
        waypoints = self._make_waypoints()
        plan = plan_trajectory(waypoints)
        arm, gripper, phase = plan.sample(plan.total_duration)
        assert arm.shape == (5,)
        # Should be near the final waypoint
        np.testing.assert_allclose(arm, waypoints[-1].arm_joints, atol=0.05)

    def test_phase_ids_monotonic(self):
        waypoints = self._make_waypoints()
        plan = plan_trajectory(waypoints)

        dt = 0.01
        t = 0.0
        prev_phase = -1
        while t <= plan.total_duration:
            _, _, phase = plan.sample(t)
            assert phase >= prev_phase, f"Phase went backward at t={t:.3f}: {prev_phase} → {phase}"
            prev_phase = phase
            t += dt

    def test_gripper_interpolation(self):
        """Gripper should interpolate between segment start/end values."""
        waypoints = self._make_waypoints()
        plan = plan_trajectory(waypoints)

        # Find the close_gripper segment (grasp→grasp_closed)
        close_seg = None
        cum_t = 0.0
        for seg in plan.segments:
            if seg.label == "grasp→grasp_closed":
                close_seg = seg
                break
            cum_t += seg.duration

        assert close_seg is not None
        # At midpoint of this segment, gripper should be between open and closed
        mid_t = cum_t + close_seg.duration / 2
        _, gripper, _ = plan.sample(mid_t)
        assert GRIPPER_CLOSE < gripper < GRIPPER_OPEN

    def test_custom_limits(self):
        waypoints = self._make_waypoints()
        limits = JointLimits(
            max_velocity=[0.5, 0.5, 0.5, 0.5, 0.75],
            max_acceleration=[1.5, 1.5, 1.5, 1.5, 2.0],
            max_jerk=[5.0, 5.0, 5.0, 5.0, 7.5],
        )
        plan = plan_trajectory(waypoints, limits)
        # Slower limits → longer duration
        default_plan = plan_trajectory(waypoints)
        assert plan.total_duration > default_plan.total_duration

    def test_requires_at_least_two_waypoints(self):
        with pytest.raises(ValueError):
            plan_trajectory([JointWaypoint(np.zeros(5), 0.0, 0, "only")])


# ---------------------------------------------------------------------------
# Integrated pipeline tests (keyframes → IK → trajectory)
# ---------------------------------------------------------------------------


class TestIntegratedPipeline:
    def test_full_pipeline(self, mj_model, mj_data, ee_site_id, arm_joint_ids):
        """End-to-end: keyframes → waypoints → trajectory."""
        home_joints = mj_data.qpos[:6].copy()
        cube_pos = np.array([0.15, 0.0, 0.395])
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
        )
        plan = plan_trajectory(waypoints)

        assert plan.total_duration > 0
        assert len(plan.segments) == 4  # 5 waypoints → 4 segments

        # Sample throughout and verify continuity
        dt = 0.01
        t = 0.0
        prev_arm = None
        while t <= plan.total_duration:
            arm, gripper, phase = plan.sample(t)
            assert arm.shape == (5,)
            if prev_arm is not None:
                # Joint positions should be continuous (no big jumps)
                delta = np.max(np.abs(arm - prev_arm))
                assert delta < 0.1, f"Joint jump at t={t:.3f}: delta={delta:.4f}"
            prev_arm = arm.copy()
            t += dt

    def test_pipeline_with_different_cube_positions(self, mj_model, mj_data, ee_site_id, arm_joint_ids):
        """Pipeline should work for various cube placements."""
        home_joints = mj_data.qpos[:6].copy()
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        cube_z = 0.395
        for cube_pos in [
            np.array([0.15, 0.0, cube_z]),
            np.array([0.18, 0.03, cube_z]),
            np.array([0.20, -0.03, cube_z]),
        ]:
            keyframes = plan_pick_keyframes(home_joints, cube_pos, cube_quat, ee_site_id, mj_model, mj_data)
            waypoints = generate_joint_waypoints(
                keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
            )
            plan = plan_trajectory(waypoints)
            assert plan.total_duration > 0
