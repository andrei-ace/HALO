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
    PHASE_VERIFY_GRASP,
)
from mujoco_sim.scene_info import TCP_PINCH_OFFSET_LOCAL, SceneInfo
from mujoco_sim.teacher.grasp_planner import GraspPose, evaluate_grasps
from mujoco_sim.teacher.keyframe_planner import plan_pick_keyframes
from mujoco_sim.teacher.trajectory import JointLimits, plan_trajectory
from mujoco_sim.teacher.trajectory_validator import get_gripper_collision_geom_ids, validate_waypoint_gripper_clearance
from mujoco_sim.teacher.waypoint_generator import JointWaypoint, generate_joint_waypoints

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCENE_XML = str(Path(__file__).resolve().parent.parent / "assets" / "so101" / "pick_scene.xml")

_DEFAULT_CUBE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def mj_model():
    return mujoco.MjModel.from_xml_path(_SCENE_XML)


@pytest.fixture(scope="module")
def scene_info(mj_model):
    return SceneInfo.from_model(mj_model)


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


def _make_side_grasp_pose(
    cube_pos: np.ndarray,
    half_sizes: np.ndarray,
    face: str = "+X",
) -> GraspPose:
    """Build a simple side-grasp GraspPose for testing keyframe planner."""
    normals = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
    }
    normal = normals[face]
    approach = -normal

    # Build orientation matrix
    up = np.array([0.0, 0.0, 1.0])
    x_ref = np.cross(up, approach)
    x_ref = x_ref / np.linalg.norm(x_ref)
    y_ref = np.cross(approach, x_ref)

    orientation = np.column_stack([x_ref, y_ref, approach])

    # Contact point = cube face center
    axis_idx = {"X": 0, "Y": 1}[face[1]]
    contact = cube_pos + normal * half_sizes[axis_idx]

    return GraspPose(
        contact_point=contact,
        orientation=orientation,
        approach_dir=approach,
        face_label=face,
        yaw_variant=0,
    )


# ---------------------------------------------------------------------------
# Keyframe planner tests
# ---------------------------------------------------------------------------


class TestKeyframePlanner:
    def test_produces_five_keyframes(self, scene_info, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)
        assert len(keyframes) == 6

    def test_keyframe_labels(self, scene_info, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)
        labels = [kf.label for kf in keyframes]
        assert labels == ["home", "pregrasp", "grasp", "grasp_closed", "lift", "verify_grasp"]

    def test_keyframe_phases(self, scene_info, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)
        phases = [kf.phase_id for kf in keyframes]
        assert phases == [
            PHASE_IDLE,
            PHASE_MOVE_PREGRASP,
            PHASE_EXECUTE_APPROACH,
            PHASE_CLOSE_GRIPPER,
            PHASE_LIFT,
            PHASE_VERIFY_GRASP,
        ]

    def test_pregrasp_along_approach(self, scene_info, mj_model, mj_data, ee_site_id):
        """Pregrasp should be offset along approach direction from contact."""
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)
        standoff = 0.15

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data, standoff=standoff)

        pregrasp = keyframes[1]
        grasp_rot = pregrasp.orientation
        offset_world = grasp_rot @ TCP_PINCH_OFFSET_LOCAL
        expected_contact = grasp_pose.contact_point - standoff * grasp_pose.approach_dir
        expected_pos = expected_contact - offset_world
        np.testing.assert_allclose(pregrasp.position, expected_pos)

    def test_grasp_at_contact_point(self, scene_info, mj_model, mj_data, ee_site_id):
        """Grasp keyframe should place jaw midpoint at contact_point."""
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)

        grasp = keyframes[2]
        grasp_rot = grasp.orientation
        offset_world = grasp_rot @ TCP_PINCH_OFFSET_LOCAL
        expected_pos = grasp_pose.contact_point - offset_world
        np.testing.assert_allclose(grasp.position, expected_pos)

    def test_gripper_sequence(self, scene_info, mj_model, mj_data, ee_site_id):
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)

        expected_grippers = [GRIPPER_OPEN, GRIPPER_OPEN, GRIPPER_OPEN, GRIPPER_CLOSE, GRIPPER_CLOSE, GRIPPER_CLOSE]
        actual = [kf.gripper for kf in keyframes]
        assert actual == expected_grippers

    def test_orientation_matches_grasp_pose(self, scene_info, mj_model, mj_data, ee_site_id):
        """Non-home keyframes should use the grasp_pose orientation."""
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)

        for kf in keyframes[1:]:
            np.testing.assert_allclose(kf.orientation, grasp_pose.orientation, atol=1e-10)

    def test_tcp_offset_applied(self, scene_info, mj_model, mj_data, ee_site_id):
        """Grasp keyframe position accounts for TCP pinch offset from contact_point."""
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)

        grasp = keyframes[2]
        grasp_rot = grasp.orientation
        jaw_midpoint = grasp.position + grasp_rot @ TCP_PINCH_OFFSET_LOCAL
        np.testing.assert_allclose(jaw_midpoint, grasp_pose.contact_point, atol=1e-10)

    def test_lift_is_vertical(self, scene_info, mj_model, mj_data, ee_site_id):
        """Lift should be vertically above the grasp contact, regardless of approach direction."""
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)
        z_lift = 0.10

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data, z_lift=z_lift)

        grasp_kf = keyframes[2]
        lift_kf = keyframes[5]
        # XY should be the same (same TCP offset, same orientation)
        np.testing.assert_allclose(lift_kf.position[:2], grasp_kf.position[:2], atol=1e-10)
        # Z should differ by z_lift
        np.testing.assert_allclose(lift_kf.position[2] - grasp_kf.position[2], z_lift, atol=1e-10)


# ---------------------------------------------------------------------------
# Waypoint generator tests
# ---------------------------------------------------------------------------


class TestWaypointGenerator:
    def test_produces_correct_count(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids):
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
        )

        assert len(waypoints) == len(keyframes)

    def test_waypoint_labels_match_keyframes(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids):
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
        )

        for kf, wp in zip(keyframes, waypoints):
            assert kf.label == wp.label
            assert kf.phase_id == wp.phase_id
            assert kf.gripper == wp.gripper

    def test_arm_joints_shape(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids):
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
        )

        for wp in waypoints:
            assert wp.arm_joints.shape == (5,)

    def test_ik_reaches_keyframe_positions(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids):
        """Verify FK of IK solution is close to keyframe target."""
        home_joints = mj_data.qpos[:6].copy()
        grasp_pose = _make_side_grasp_pose(scene_info.green_cube_default_pos, scene_info.green_cube_half_sizes)

        keyframes = plan_pick_keyframes(home_joints, grasp_pose, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints=home_joints[:5]
        )

        d_check = mujoco.MjData(mj_model)
        for kf, wp in zip(keyframes, waypoints):
            d_check.qpos[:] = mj_data.qpos[:]
            for i, jid in enumerate(arm_joint_ids):
                d_check.qpos[mj_model.jnt_qposadr[jid]] = wp.arm_joints[i]
            mujoco.mj_forward(mj_model, d_check)
            ee_pos = d_check.site_xpos[ee_site_id]
            err = float(np.linalg.norm(kf.position - ee_pos))
            assert err < 0.015, f"IK error for '{kf.label}': {err:.4f} m"


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
            max_velocity=[0.2, 0.2, 0.2, 0.2, 0.3],
            max_acceleration=[0.5, 0.5, 0.5, 0.5, 0.7],
            max_jerk=[1.0, 1.0, 1.0, 1.0, 1.5],
        )
        plan = plan_trajectory(waypoints, limits)
        # Slower limits → longer duration
        default_plan = plan_trajectory(waypoints)
        assert plan.total_duration > default_plan.total_duration

    def test_requires_at_least_two_waypoints(self):
        with pytest.raises(ValueError):
            plan_trajectory([JointWaypoint(np.zeros(5), 0.0, 0, "only")])


# ---------------------------------------------------------------------------
# Integrated pipeline tests (grasp_planner → keyframes → IK → trajectory)
# ---------------------------------------------------------------------------


class TestIntegratedPipeline:
    def test_full_pipeline(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids):
        """End-to-end: evaluate_grasps → keyframes → waypoints → trajectory."""
        home_joints = mj_data.qpos[:6].copy()

        best = evaluate_grasps(
            scene_info.green_cube_default_pos,
            _DEFAULT_CUBE_QUAT,
            scene_info.green_cube_half_sizes,
            mj_model,
            mj_data,
            ee_site_id,
            arm_joint_ids,
            home_joints[:5],
            table_z=scene_info.table_z,
        )

        keyframes = plan_pick_keyframes(home_joints, best.grasp, ee_site_id, mj_model, mj_data)
        waypoints = generate_joint_waypoints(
            keyframes,
            mj_model,
            mj_data,
            ee_site_id,
            arm_joint_ids,
            seed_joints=home_joints[:5],
            pos_tol=0.03,
        )
        plan = plan_trajectory(waypoints)

        assert plan.total_duration > 0
        assert len(plan.segments) == 5  # 6 waypoints → 5 segments

        # Sample throughout and verify continuity
        dt = 0.01
        t = 0.0
        prev_arm = None
        while t <= plan.total_duration:
            arm, gripper, phase = plan.sample(t)
            assert arm.shape == (5,)
            if prev_arm is not None:
                delta = np.max(np.abs(arm - prev_arm))
                assert delta < 0.1, f"Joint jump at t={t:.3f}: delta={delta:.4f}"
            prev_arm = arm.copy()
            t += dt

    def test_pipeline_with_different_cube_positions(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids):
        """Pipeline should work for various cube placements."""
        home_joints = mj_data.qpos[:6].copy()

        cx, cy, cube_z = scene_info.green_cube_default_pos
        for cube_pos in [
            np.array([cx, cy, cube_z]),
            np.array([cx + 0.01, cy + 0.01, cube_z]),
            np.array([cx - 0.01, cy - 0.01, cube_z]),
        ]:
            best = evaluate_grasps(
                cube_pos,
                _DEFAULT_CUBE_QUAT,
                scene_info.green_cube_half_sizes,
                mj_model,
                mj_data,
                ee_site_id,
                arm_joint_ids,
                home_joints[:5],
                table_z=scene_info.table_z,
            )

            keyframes = plan_pick_keyframes(home_joints, best.grasp, ee_site_id, mj_model, mj_data)
            waypoints = generate_joint_waypoints(
                keyframes,
                mj_model,
                mj_data,
                ee_site_id,
                arm_joint_ids,
                seed_joints=home_joints[:5],
                pos_tol=0.03,
            )
            plan = plan_trajectory(waypoints)
            assert plan.total_duration > 0


# ---------------------------------------------------------------------------
# Place keyframe planner — tray target
# ---------------------------------------------------------------------------


class TestPlaceTrayCandidate:
    """Tests for plan_place_candidates with target_body='tray'."""

    def test_tray_single_candidate(self, mj_model, mj_data, ee_site_id, scene_info):
        from mujoco_sim.teacher.place_keyframe_planner import plan_place_candidates

        home = np.zeros(6)
        held_half = scene_info.green_cube_half_sizes

        candidates = plan_place_candidates(
            current_joints=home,
            reference_pos=scene_info.tray_pos,
            ee_site_id=ee_site_id,
            model=mj_model,
            data=mj_data,
            table_z=scene_info.table_z,
            held_half_sizes=held_half,
            ref_half_sizes=np.zeros(3),
            target_body="tray",
            tray_floor_z=scene_info.tray_floor_z,
        )

        assert len(candidates) == 1
        keyframes = candidates[0]
        assert len(keyframes) == 5

    def test_tray_place_z_uses_tray_floor(self, mj_model, mj_data, ee_site_id, scene_info):
        from mujoco_sim.teacher.place_keyframe_planner import plan_place_candidates

        home = np.zeros(6)
        held_half = scene_info.green_cube_half_sizes

        candidates = plan_place_candidates(
            current_joints=home,
            reference_pos=scene_info.tray_pos,
            ee_site_id=ee_site_id,
            model=mj_model,
            data=mj_data,
            table_z=scene_info.table_z,
            held_half_sizes=held_half,
            ref_half_sizes=np.zeros(3),
            target_body="tray",
            tray_floor_z=scene_info.tray_floor_z,
        )

        # Place keyframe (index 2) Z should be tray_floor_z + held_half_z
        place_kf = candidates[0][2]
        expected_z = scene_info.tray_floor_z + held_half[2]
        assert abs(place_kf.position[2] - expected_z) < 1e-6

    def test_tray_place_xy_centered(self, mj_model, mj_data, ee_site_id, scene_info):
        from mujoco_sim.teacher.place_keyframe_planner import plan_place_candidates

        home = np.zeros(6)
        held_half = scene_info.green_cube_half_sizes

        candidates = plan_place_candidates(
            current_joints=home,
            reference_pos=scene_info.tray_pos,
            ee_site_id=ee_site_id,
            model=mj_model,
            data=mj_data,
            table_z=scene_info.table_z,
            held_half_sizes=held_half,
            ref_half_sizes=np.zeros(3),
            target_body="tray",
            tray_floor_z=scene_info.tray_floor_z,
        )

        place_kf = candidates[0][2]
        assert abs(place_kf.position[0] - scene_info.tray_pos[0]) < 1e-6
        assert abs(place_kf.position[1] - scene_info.tray_pos[1]) < 1e-6


# ---------------------------------------------------------------------------
# Waypoint gripper clearance tests
# ---------------------------------------------------------------------------


class TestWaypointGripperClearance:
    def test_get_gripper_collision_geom_ids(self, mj_model):
        """get_gripper_collision_geom_ids returns non-empty list of valid IDs."""
        geom_ids = get_gripper_collision_geom_ids(mj_model)
        assert len(geom_ids) > 0
        for gid in geom_ids:
            assert 0 <= gid < mj_model.ngeom

    def test_waypoint_gripper_clearance_above_table(self, mj_model, mj_data, ee_site_id, arm_joint_ids, scene_info):
        """Home-like waypoints well above table should pass clearance."""
        home_joints = mj_data.qpos[:5].copy()
        gripper_geom_ids = get_gripper_collision_geom_ids(mj_model)

        waypoints = [
            JointWaypoint(home_joints, GRIPPER_OPEN, PHASE_IDLE, "home"),
            JointWaypoint(home_joints.copy(), GRIPPER_CLOSE, PHASE_MOVE_PREGRASP, "home_closed"),
        ]

        result = validate_waypoint_gripper_clearance(
            waypoints,
            mj_model,
            mj_data,
            arm_joint_ids,
            gripper_geom_ids,
            ee_site_id,
            scene_info.table_z,
            margin=0.01,
        )
        assert result is True

    def test_waypoint_gripper_clearance_below_table(self, mj_model, mj_data, ee_site_id, arm_joint_ids, scene_info):
        """Waypoints with arm forced below table Z should fail clearance."""
        gripper_geom_ids = get_gripper_collision_geom_ids(mj_model)

        # Joint config that puts EE ~0.349 and gripper geoms ~0.361, both below threshold 0.38.
        bad_joints = np.array([0.0, 0.5, 1.0, 1.0, 0.0])

        waypoints = [
            JointWaypoint(bad_joints, GRIPPER_OPEN, PHASE_EXECUTE_APPROACH, "below_table"),
        ]

        result = validate_waypoint_gripper_clearance(
            waypoints,
            mj_model,
            mj_data,
            arm_joint_ids,
            gripper_geom_ids,
            ee_site_id,
            scene_info.table_z,
            margin=0.01,
        )
        assert result is False
