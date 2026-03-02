"""Tests for the multi-face grasp planner.

Uses a real MuJoCo model (SO-101 pick_scene.xml) for IK-dependent tests.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_sim.scene_info import SceneInfo
from mujoco_sim.teacher.grasp_planner import (
    GraspPlanningFailure,
    GraspPose,
    ScoredGrasp,
    enumerate_face_grasps,
    evaluate_grasps,
    filter_grasps,
    score_grasp,
)

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


@pytest.fixture()
def seed_joints(mj_data):
    return mj_data.qpos[:5].copy()


# ---------------------------------------------------------------------------
# enumerate_face_grasps tests
# ---------------------------------------------------------------------------


class TestEnumerateFaceGrasps:
    def test_enumerate_produces_64_candidates(self, scene_info):
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes
        )
        assert len(candidates) == 64  # 4 faces × 16 per face

    def test_enumerate_custom_count(self, scene_info):
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes, n_candidates=32
        )
        assert len(candidates) == 32

    def test_enumerate_face_labels(self, scene_info):
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes
        )
        labels = [c.face_label for c in candidates]
        for face in ["+X", "-X", "+Y", "-Y"]:
            assert labels.count(face) == 16  # 64 / 4

    def test_enumerate_rotated_cube(self, scene_info):
        """Non-identity quaternion should shift face normals."""
        identity_candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes, seed=0
        )

        # 45-degree yaw rotation
        angle = np.pi / 4
        rotated_quat = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        rotated_candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, rotated_quat, scene_info.green_cube_half_sizes, seed=0
        )

        assert len(rotated_candidates) == 64
        # Contact points should differ between identity and rotated
        id_contacts = np.array([c.contact_point for c in identity_candidates])
        rot_contacts = np.array([c.contact_point for c in rotated_candidates])
        assert not np.allclose(id_contacts, rot_contacts, atol=1e-3)

    def test_approach_is_unit_vector(self, scene_info):
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes
        )
        for c in candidates:
            np.testing.assert_allclose(np.linalg.norm(c.approach_dir), 1.0, atol=1e-10)

    def test_orientation_is_valid_rotation(self, scene_info):
        """Each orientation should be a proper rotation matrix (det=+1, orthogonal)."""
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes
        )
        for c in candidates:
            assert c.orientation.shape == (3, 3)
            np.testing.assert_allclose(c.orientation.T @ c.orientation, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(c.orientation), 1.0, atol=1e-10)

    def test_z_column_equals_approach(self, scene_info):
        """Gripperframe Z-column should equal approach_dir."""
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes
        )
        for c in candidates:
            np.testing.assert_allclose(c.orientation[:, 2], c.approach_dir, atol=1e-10)

    def test_approach_within_cone(self, scene_info):
        """All approach directions should be within max_cone_deg of the face normal."""
        max_cone = 5.0
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos,
            _DEFAULT_CUBE_QUAT,
            scene_info.green_cube_half_sizes,
            max_cone_deg=max_cone,
        )
        face_normals = {"+X": [1, 0, 0], "-X": [-1, 0, 0], "+Y": [0, 1, 0], "-Y": [0, -1, 0]}
        for c in candidates:
            normal = np.array(face_normals[c.face_label])
            # approach_dir ≈ -normal (into the face), so angle with -normal should be ≤ cone
            cos_a = np.dot(c.approach_dir, -normal)
            angle_deg = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            assert angle_deg <= max_cone + 0.01, (
                f"{c.face_label} approach deviates {angle_deg:.2f}° > {max_cone}° from face normal"
            )

    def test_seed_reproducibility(self, scene_info):
        """Same seed should produce identical candidates."""
        a = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes, seed=42
        )
        b = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes, seed=42
        )
        for ca, cb in zip(a, b):
            np.testing.assert_array_equal(ca.approach_dir, cb.approach_dir)
            np.testing.assert_array_equal(ca.orientation, cb.orientation)

    def test_face_contact_span_controls_center_vs_offcenter(self, scene_info):
        face_normals = {
            "+X": np.array([1.0, 0.0, 0.0]),
            "-X": np.array([-1.0, 0.0, 0.0]),
            "+Y": np.array([0.0, 1.0, 0.0]),
            "-Y": np.array([0.0, -1.0, 0.0]),
        }

        center_only = enumerate_face_grasps(
            scene_info.green_cube_default_pos,
            _DEFAULT_CUBE_QUAT,
            scene_info.green_cube_half_sizes,
            face_contact_span=0.0,
            face_standoff=0.0,
            seed=123,
        )
        for c in center_only:
            axis_idx = {"X": 0, "Y": 1, "Z": 2}[c.face_label[1]]
            expected = (
                scene_info.green_cube_default_pos
                + face_normals[c.face_label] * scene_info.green_cube_half_sizes[axis_idx]
            )
            np.testing.assert_allclose(c.contact_point, expected, atol=1e-10)

        offcenter = enumerate_face_grasps(
            scene_info.green_cube_default_pos,
            _DEFAULT_CUBE_QUAT,
            scene_info.green_cube_half_sizes,
            face_contact_span=0.35,
            face_standoff=0.0,
            seed=123,
        )
        assert any(np.linalg.norm(a.contact_point - b.contact_point) > 1e-8 for a, b in zip(center_only, offcenter)), (
            "Expected at least one off-center contact point when face_contact_span > 0"
        )


# ---------------------------------------------------------------------------
# filter_grasps tests
# ---------------------------------------------------------------------------


class TestFilterGrasps:
    def test_filter_removes_below_table(self, scene_info):
        """A candidate with approach pointing downward should be filtered."""
        # Craft a fake candidate where pregrasp is below table
        grasp = GraspPose(
            contact_point=np.array([0.15, 0.0, 0.35]),  # below table
            orientation=np.eye(3),
            approach_dir=np.array([0.0, 0.0, -1.0]),  # pointing down
            face_label="+Z",
            yaw_variant=0,
        )
        result = filter_grasps([grasp], table_z=scene_info.table_z)
        assert len(result) == 0

    def test_filter_keeps_side_grasps(self, scene_info):
        """Default cube position side faces should pass."""
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes
        )
        feasible = filter_grasps(candidates, table_z=scene_info.table_z)
        # At least some side grasps should pass for the default cube position
        assert len(feasible) > 0


# ---------------------------------------------------------------------------
# score_grasp tests
# ---------------------------------------------------------------------------


class TestScoreGrasp:
    def test_score_returns_none_for_unreachable(
        self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints
    ):
        """Cube far out of reach should return None."""
        far_cube = np.array([1.0, 0.0, scene_info.green_cube_default_pos[2]])
        candidates = enumerate_face_grasps(far_cube, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes)
        feasible = filter_grasps(candidates, table_z=scene_info.table_z)
        # Score each — all should fail
        for candidate in feasible:
            result = score_grasp(candidate, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints)
            assert result is None

    def test_score_returns_scored_grasp(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints):
        """Default cube position should have at least one scoreable candidate."""
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes
        )
        feasible = filter_grasps(candidates, table_z=scene_info.table_z)

        scored_any = False
        for candidate in feasible:
            result = score_grasp(candidate, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints)
            if result is not None:
                scored_any = True
                assert result.grasp_joints.shape == (5,)
                assert result.pregrasp_joints.shape == (5,)
                assert 0.0 <= result.score <= 1.0
                break
        assert scored_any, "Expected at least one successful grasp score"

    def test_ori_error_within_tolerance(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints):
        """Scored grasps should have orientation error within tolerance."""
        candidates = enumerate_face_grasps(
            scene_info.green_cube_default_pos, _DEFAULT_CUBE_QUAT, scene_info.green_cube_half_sizes
        )
        feasible = filter_grasps(candidates, table_z=scene_info.table_z)

        for candidate in feasible:
            result = score_grasp(candidate, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints, ori_tol_deg=55.0)
            if result is not None:
                assert result.ori_err_deg <= 55.0


# ---------------------------------------------------------------------------
# evaluate_grasps tests
# ---------------------------------------------------------------------------


class TestEvaluateGrasps:
    def test_evaluate_returns_best(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints):
        """Returned grasp should have the best score."""
        best = evaluate_grasps(
            scene_info.green_cube_default_pos,
            _DEFAULT_CUBE_QUAT,
            scene_info.green_cube_half_sizes,
            mj_model,
            mj_data,
            ee_site_id,
            arm_joint_ids,
            seed_joints,
            table_z=scene_info.table_z,
        )
        assert isinstance(best, ScoredGrasp)
        assert best.score > 0.0

    def test_evaluate_default_cube_pos(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints):
        """evaluate_grasps works for the default cube position."""
        best = evaluate_grasps(
            scene_info.green_cube_default_pos,
            _DEFAULT_CUBE_QUAT,
            scene_info.green_cube_half_sizes,
            mj_model,
            mj_data,
            ee_site_id,
            arm_joint_ids,
            seed_joints,
            table_z=scene_info.table_z,
        )
        assert best.grasp.face_label in ["+X", "-X", "+Y", "-Y"]
        assert best.ik_pos_err <= 0.02
        assert best.ori_err_deg <= 55.0

    def test_evaluate_various_positions(self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints):
        """evaluate_grasps works for several cube placements."""
        cube_z = scene_info.green_cube_default_pos[2]
        cx, cy = scene_info.green_cube_default_pos[0], scene_info.green_cube_default_pos[1]
        positions = [
            np.array([cx, cy, cube_z]),
            np.array([cx + 0.01, cy + 0.01, cube_z]),
            np.array([cx - 0.01, cy - 0.01, cube_z]),
        ]
        for pos in positions:
            best = evaluate_grasps(
                pos,
                _DEFAULT_CUBE_QUAT,
                scene_info.green_cube_half_sizes,
                mj_model,
                mj_data,
                ee_site_id,
                arm_joint_ids,
                seed_joints,
                table_z=scene_info.table_z,
            )
            assert best.score > 0.0, f"No grasp found for cube at {pos}"

    def test_evaluate_raises_on_unreachable(
        self, scene_info, mj_model, mj_data, ee_site_id, arm_joint_ids, seed_joints
    ):
        """Cube far out of reach should raise GraspPlanningFailure."""
        far_pos = np.array([1.0, 0.0, scene_info.green_cube_default_pos[2]])
        with pytest.raises(GraspPlanningFailure):
            evaluate_grasps(
                far_pos,
                _DEFAULT_CUBE_QUAT,
                scene_info.green_cube_half_sizes,
                mj_model,
                mj_data,
                ee_site_id,
                arm_joint_ids,
                seed_joints,
                table_z=scene_info.table_z,
            )
