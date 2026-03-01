"""Multi-face grasp planner for parallel-jaw pick on a cube.

Generates 64 random grasp candidates across 4 side faces (16 per face), each
with an approach direction within a tight cone (5°) of the face normal and a
random yaw.  Filters geometrically infeasible ones (table collision, unreachable
pregrasp), scores the rest by IK quality + joint-limit margin + manipulability +
orientation error, and returns the best candidate.

Pipeline::

    enumerate_face_grasps  →  filter_grasps  →  score_grasp (each)  →  best ScoredGrasp

Gripperframe axes convention:
    Z = approach direction (into the face)
    X = jaw hinge direction
    Y = jaw span (lateral) direction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_sim.scene_info import TCP_PINCH_OFFSET_LOCAL
from mujoco_sim.teacher.ik_helper import solve_ik, solve_ik_with_orientation

logger = logging.getLogger(__name__)


@dataclass
class GraspPose:
    """A candidate grasp pose for a cube face."""

    contact_point: np.ndarray  # (3,) world — where jaw midpoint should land (face center)
    orientation: np.ndarray  # (3,3) rotation matrix for gripperframe
    approach_dir: np.ndarray  # (3,) unit vector (gripperframe Z = approach into face)
    face_label: str  # "+X", "-X", "+Y", "-Y"
    yaw_variant: int  # 0 or 1
    tilt_deg: float = 0.0  # downward tilt from horizontal (0=horizontal, 30=angled down)


@dataclass
class ScoredGrasp:
    """A grasp candidate with IK solution and quality score."""

    grasp: GraspPose
    grasp_joints: np.ndarray  # (5,) IK solution at grasp pose
    pregrasp_joints: np.ndarray  # (5,) IK solution at pregrasp
    score: float  # combined 0–1
    ik_pos_err: float  # metres
    ori_err_deg: float  # degrees
    joint_margin: float  # 0–1
    manipulability: float  # 0–1


class GraspPlanningFailure(Exception):
    """Raised when no feasible grasp candidate passes all checks."""


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to (3,3) rotation matrix."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def _build_gripper_rotation(approach: np.ndarray, yaw_angle: float) -> np.ndarray:
    """Build a gripperframe rotation matrix for a given approach direction and yaw.

    Gripperframe columns: [x_grip, y_grip, approach] where:
        Z = approach (into the face)
        X = jaw hinge direction
        Y = jaw span direction

    Args:
        approach: (3,) unit vector — gripperframe Z-axis (points into face).
        yaw_angle: Rotation about approach axis (0 or pi/2 for the two variants).

    Returns:
        (3,3) rotation matrix with det=+1.
    """
    # Pick a reference up vector not parallel to approach
    up = np.array([0.0, 0.0, 1.0]) if abs(approach[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_ref = _normalize(np.cross(up, approach))
    y_ref = np.cross(approach, x_ref)

    # Apply yaw variant (rotation about approach axis)
    c, s = np.cos(yaw_angle), np.sin(yaw_angle)
    x_grip = c * x_ref + s * y_ref
    y_grip = -s * x_ref + c * y_ref

    return np.column_stack([x_grip, y_grip, approach])


# Side face normals in cube-local frame (4 faces, no top/bottom)
_SIDE_FACE_NORMALS: list[tuple[str, np.ndarray]] = [
    ("+X", np.array([1.0, 0.0, 0.0])),
    ("-X", np.array([-1.0, 0.0, 0.0])),
    ("+Y", np.array([0.0, 1.0, 0.0])),
    ("-Y", np.array([0.0, -1.0, 0.0])),
]


def _random_cone_vector(axis: np.ndarray, max_half_angle_rad: float, rng: np.random.RandomState) -> np.ndarray:
    """Sample a unit vector uniformly within a cone around *axis*.

    Args:
        axis: (3,) unit vector — cone centre.
        max_half_angle_rad: Half-angle of the cone in radians.
        rng: NumPy random state.

    Returns:
        (3,) unit vector within the cone.
    """
    # Uniform azimuth, cos-uniform polar within [0, max_half_angle]
    phi = rng.uniform(0, 2 * np.pi)
    cos_theta_min = np.cos(max_half_angle_rad)
    cos_theta = rng.uniform(cos_theta_min, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

    # Local frame: z = axis
    up = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _normalize(np.cross(up, axis))
    v = np.cross(axis, u)

    return _normalize(axis * cos_theta + u * sin_theta * np.cos(phi) + v * sin_theta * np.sin(phi))


def enumerate_face_grasps(
    cube_pos: np.ndarray,
    cube_quat: np.ndarray,
    cube_half_sizes: np.ndarray,
    *,
    n_candidates: int = 64,
    max_cone_deg: float = 5.0,
    seed: int | None = None,
) -> list[GraspPose]:
    """Enumerate random side-face grasp candidates for a cube.

    For each of the 4 side faces, generates ``n_candidates // 4`` random
    approach directions within a narrow cone (``max_cone_deg``) around the
    face normal, each with a random yaw.  The contact point is always the
    face centre.

    Args:
        cube_pos: (3,) cube center in world frame.
        cube_quat: (4,) cube quaternion [w, x, y, z].
        cube_half_sizes: (3,) half-extents [hx, hy, hz].
        n_candidates: Total number of candidates (split evenly across 4 faces).
        max_cone_deg: Maximum angular deviation from face normal (degrees).
        seed: Random seed for reproducibility (None = random).

    Returns:
        List of GraspPose candidates.
    """
    rng = np.random.RandomState(seed)
    R_cube = _quat_to_rotmat(cube_quat)
    candidates: list[GraspPose] = []
    n_per_face = n_candidates // len(_SIDE_FACE_NORMALS)
    max_cone_rad = np.radians(max_cone_deg)

    for face_label, normal_local in _SIDE_FACE_NORMALS:
        normal_world = R_cube @ normal_local

        # Contact point = face centre in world frame
        axis_idx = {"X": 0, "Y": 1, "Z": 2}[face_label[1]]
        contact_point = cube_pos + normal_world * cube_half_sizes[axis_idx]

        # Base approach direction: gripper Z points INTO the face
        approach_centre = -normal_world

        for i in range(n_per_face):
            approach_dir = _random_cone_vector(approach_centre, max_cone_rad, rng)
            yaw_angle = rng.uniform(0, 2 * np.pi)
            orientation = _build_gripper_rotation(approach_dir, yaw_angle)

            # Compute tilt from horizontal for diagnostics
            tilt_deg = float(np.degrees(np.arcsin(np.clip(-approach_dir[2], -1.0, 1.0))))

            candidates.append(
                GraspPose(
                    contact_point=contact_point.copy(),
                    orientation=orientation,
                    approach_dir=approach_dir.copy(),
                    face_label=face_label,
                    yaw_variant=i,
                    tilt_deg=tilt_deg,
                )
            )

    return candidates


def filter_grasps(
    candidates: list[GraspPose],
    *,
    table_z: float,
    standoff: float = 0.04,
    gripper_depth: float = 0.10,
    margin: float = 0.01,
) -> list[GraspPose]:
    """Filter out geometrically infeasible grasp candidates.

    Rejects candidates where:
    - Pregrasp point is below table_z + margin.
    - Gripper body at grasp extends below table surface.

    Args:
        candidates: List of GraspPose to filter.
        table_z: Table surface height in world frame.
        standoff: Distance to back off along approach direction for pregrasp.
        gripper_depth: Approximate depth of gripper body behind contact point.
        margin: Safety margin above table.

    Returns:
        Filtered list of feasible GraspPose candidates.
    """
    feasible: list[GraspPose] = []
    threshold = table_z + margin

    for grasp in candidates:
        # Check pregrasp point
        pregrasp_point = grasp.contact_point - standoff * grasp.approach_dir
        if pregrasp_point[2] < threshold:
            logger.debug(
                "Filtered %s yaw=%d: pregrasp below table (z=%.3f)",
                grasp.face_label,
                grasp.yaw_variant,
                pregrasp_point[2],
            )
            continue

        # Check gripper body at grasp pose — the gripper extends behind
        # the contact along negative approach direction
        gripper_back = grasp.contact_point - gripper_depth * grasp.approach_dir
        min_z = min(grasp.contact_point[2], gripper_back[2])
        if min_z < threshold:
            logger.debug(
                "Filtered %s yaw=%d: gripper body below table (min_z=%.3f)",
                grasp.face_label,
                grasp.yaw_variant,
                min_z,
            )
            continue

        feasible.append(grasp)

    return feasible


def score_grasp(
    grasp: GraspPose,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    arm_joint_ids: list[int],
    seed_joints: np.ndarray,
    *,
    standoff: float = 0.04,
    z_lift: float = 0.08,
    tcp_offset: np.ndarray | None = None,
    pos_tol: float = 0.01,
    ori_tol_deg: float = 55.0,
) -> ScoredGrasp | None:
    """Score a single grasp candidate by solving IK and evaluating quality.

    Returns None if IK fails or tolerances are exceeded.

    Args:
        grasp: Candidate grasp pose.
        model: MuJoCo model.
        data: MuJoCo data (not mutated).
        ee_site_id: MuJoCo site id for gripperframe.
        arm_joint_ids: List of 5 arm joint qpos indices.
        seed_joints: (5,) seed joint configuration for IK.
        standoff: Distance along approach_dir for pregrasp offset.
        z_lift: Vertical lift height above grasp for lift-reachability check.
        tcp_offset: (3,) local-frame TCP offset; defaults to TCP_PINCH_OFFSET_LOCAL.
        pos_tol: Maximum acceptable position error (metres).
        ori_tol_deg: Maximum acceptable orientation error (degrees).

    Returns:
        ScoredGrasp or None if infeasible.
    """
    if tcp_offset is None:
        tcp_offset = TCP_PINCH_OFFSET_LOCAL

    # Compute IK targets (site position = contact - rot @ tcp_offset)
    grasp_site_target = grasp.contact_point - grasp.orientation @ tcp_offset
    pregrasp_contact = grasp.contact_point - standoff * grasp.approach_dir
    pregrasp_site_target = pregrasp_contact - grasp.orientation @ tcp_offset

    # Solve IK for grasp pose
    grasp_joints = solve_ik_with_orientation(
        model,
        data,
        grasp_site_target,
        grasp.orientation,
        ee_site_id,
        arm_joint_ids,
        pos_weight=1.0,
        ori_weight=0.3,
        max_iters=300,
        tol=1e-3,
        damping=1e-2,
    )

    # FK-verify grasp position
    d_check = mujoco.MjData(model)
    d_check.qpos[:] = data.qpos[:]
    for i, jid in enumerate(arm_joint_ids):
        d_check.qpos[jid] = grasp_joints[i]
    mujoco.mj_forward(model, d_check)

    ee_pos = d_check.site_xpos[ee_site_id].copy()
    ik_pos_err = float(np.linalg.norm(grasp_site_target - ee_pos))
    if ik_pos_err > pos_tol:
        logger.debug(
            "Rejected %s yaw=%d: pos_err=%.4f > %.4f",
            grasp.face_label,
            grasp.yaw_variant,
            ik_pos_err,
            pos_tol,
        )
        return None

    # Compute orientation error (angle between achieved and target Z-axes)
    achieved_rot = d_check.site_xmat[ee_site_id].reshape(3, 3)
    achieved_z = achieved_rot[:, 2]
    target_z = grasp.orientation[:, 2]
    cos_angle = np.clip(np.dot(achieved_z, target_z), -1.0, 1.0)
    ori_err_deg = float(np.degrees(np.arccos(cos_angle)))

    if ori_err_deg > ori_tol_deg:
        logger.debug(
            "Rejected %s yaw=%d: ori_err=%.1f° > %.1f°",
            grasp.face_label,
            grasp.yaw_variant,
            ori_err_deg,
            ori_tol_deg,
        )
        return None

    # Solve IK for pregrasp pose (seed from grasp solution)
    d_seed = mujoco.MjData(model)
    d_seed.qpos[:] = data.qpos[:]
    for i, jid in enumerate(arm_joint_ids):
        d_seed.qpos[jid] = grasp_joints[i]
    mujoco.mj_forward(model, d_seed)

    pregrasp_joints = solve_ik_with_orientation(
        model,
        d_seed,
        pregrasp_site_target,
        grasp.orientation,
        ee_site_id,
        arm_joint_ids,
        pos_weight=1.0,
        ori_weight=0.3,
        max_iters=300,
        tol=1e-3,
        damping=1e-2,
    )

    # FK-verify pregrasp position
    for i, jid in enumerate(arm_joint_ids):
        d_check.qpos[jid] = pregrasp_joints[i]
    mujoco.mj_forward(model, d_check)
    pregrasp_pos_err = float(np.linalg.norm(pregrasp_site_target - d_check.site_xpos[ee_site_id]))
    if pregrasp_pos_err > pos_tol:
        logger.debug(
            "Rejected %s yaw=%d: pregrasp pos_err=%.4f > %.4f",
            grasp.face_label,
            grasp.yaw_variant,
            pregrasp_pos_err,
            pos_tol,
        )
        return None

    # Solve IK for lift pose (position-only — orientation can relax during lift
    # since the object is already grasped; seed from grasp solution)
    lift_contact = grasp.contact_point + np.array([0.0, 0.0, z_lift])
    lift_site_target = lift_contact - grasp.orientation @ tcp_offset
    for i, jid in enumerate(arm_joint_ids):
        d_seed.qpos[jid] = grasp_joints[i]
    mujoco.mj_forward(model, d_seed)

    lift_joints = solve_ik(
        model,
        d_seed,
        lift_site_target,
        ee_site_id,
        arm_joint_ids,
        max_iters=200,
        tol=1e-3,
    )

    # FK-verify lift position
    for i, jid in enumerate(arm_joint_ids):
        d_check.qpos[jid] = lift_joints[i]
    mujoco.mj_forward(model, d_check)
    lift_pos_err = float(np.linalg.norm(lift_site_target - d_check.site_xpos[ee_site_id]))
    lift_pos_tol = max(pos_tol, 0.03)  # relaxed for lift (object already grasped)
    if lift_pos_err > lift_pos_tol:
        logger.debug(
            "Rejected %s yaw=%d: lift pos_err=%.4f > %.4f",
            grasp.face_label,
            grasp.yaw_variant,
            lift_pos_err,
            lift_pos_tol,
        )
        return None

    # Compute joint-limit margin: min over joints of (1 - |q - mid| / half_range)
    joint_margins = []
    for i, jid in enumerate(arm_joint_ids):
        lo, hi = model.jnt_range[jid]
        mid = (lo + hi) / 2
        half_range = (hi - lo) / 2
        if half_range > 0:
            margin = 1.0 - abs(grasp_joints[i] - mid) / half_range
        else:
            margin = 0.0
        joint_margins.append(margin)
    joint_margin = float(min(joint_margins))

    # Compute manipulability: sqrt(det(J @ J.T)) on 3×5 position Jacobian
    # (reuse d_check which has grasp_joints loaded)
    for i, jid in enumerate(arm_joint_ids):
        d_check.qpos[jid] = grasp_joints[i]
    mujoco.mj_forward(model, d_check)

    jac_pos = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, d_check, jac_pos, None, ee_site_id)
    J = jac_pos[:, arm_joint_ids]  # (3, 5)
    manip_raw = float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    # Normalize manipulability to 0–1 range (typical range ~0 to ~0.01 for this arm)
    manip_scale = 0.005  # empirical: good configs around 0.002-0.005
    manipulability = min(1.0, manip_raw / manip_scale)

    # Combined score
    score = (
        0.30 * (1.0 - ik_pos_err / pos_tol)
        + 0.25 * max(0.0, joint_margin)
        + 0.25 * (1.0 - ori_err_deg / ori_tol_deg)
        + 0.20 * manipulability
    )

    return ScoredGrasp(
        grasp=grasp,
        grasp_joints=grasp_joints,
        pregrasp_joints=pregrasp_joints,
        score=score,
        ik_pos_err=ik_pos_err,
        ori_err_deg=ori_err_deg,
        joint_margin=joint_margin,
        manipulability=manipulability,
    )


def evaluate_grasps(
    cube_pos: np.ndarray,
    cube_quat: np.ndarray,
    cube_half_sizes: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    arm_joint_ids: list[int],
    seed_joints: np.ndarray,
    *,
    standoff: float = 0.04,
    z_lift: float = 0.08,
    table_z: float,
    tcp_offset: np.ndarray | None = None,
    pos_tol: float = 0.01,
    ori_tol_deg: float = 55.0,
    best_effort: bool = False,
) -> ScoredGrasp:
    """Enumerate, filter, score, and select the best grasp candidate.

    Args:
        cube_pos: (3,) cube center in world frame.
        cube_quat: (4,) cube quaternion [w, x, y, z].
        cube_half_sizes: (3,) half-extents [hx, hy, hz].
        model: MuJoCo model.
        data: MuJoCo data (not mutated).
        ee_site_id: MuJoCo site id for gripperframe.
        arm_joint_ids: List of 5 arm joint qpos indices.
        seed_joints: (5,) seed joint configuration for IK.
        standoff: Pregrasp standoff distance along approach direction.
        z_lift: Vertical lift height for lift-reachability check.
        table_z: Table surface height.
        tcp_offset: (3,) local-frame TCP offset.
        pos_tol: IK position tolerance (metres).
        ori_tol_deg: Orientation tolerance (degrees).
        best_effort: If True, retry with very relaxed tolerances instead of raising.

    Returns:
        Best ScoredGrasp.

    Raises:
        GraspPlanningFailure: If no candidate passes all checks (and best_effort is False).
    """
    candidates = enumerate_face_grasps(cube_pos, cube_quat, cube_half_sizes)
    logger.info("Enumerated %d grasp candidates", len(candidates))

    feasible = filter_grasps(candidates, table_z=table_z, standoff=standoff)
    logger.info("After geometric filter: %d feasible candidates", len(feasible))

    if not feasible:
        raise GraspPlanningFailure(f"All {len(candidates)} candidates filtered by geometry")

    scored: list[ScoredGrasp] = []
    for candidate in feasible:
        result = score_grasp(
            candidate,
            model,
            data,
            ee_site_id,
            arm_joint_ids,
            seed_joints,
            standoff=standoff,
            z_lift=z_lift,
            tcp_offset=tcp_offset,
            pos_tol=pos_tol,
            ori_tol_deg=ori_tol_deg,
        )
        if result is not None:
            scored.append(result)
            logger.info(
                "  %s yaw=%d tilt=%.0f°: score=%.3f (pos_err=%.4f m, ori_err=%.1f°, margin=%.2f, manip=%.3f)",
                candidate.face_label,
                candidate.yaw_variant,
                candidate.tilt_deg,
                result.score,
                result.ik_pos_err,
                result.ori_err_deg,
                result.joint_margin,
                result.manipulability,
            )

    if not scored and best_effort:
        logger.warning("No candidates passed strict scoring — retrying with relaxed tolerances (best_effort)")
        for candidate in feasible:
            result = score_grasp(
                candidate,
                model,
                data,
                ee_site_id,
                arm_joint_ids,
                seed_joints,
                standoff=standoff,
                z_lift=z_lift,
                tcp_offset=tcp_offset,
                pos_tol=0.10,
                ori_tol_deg=180.0,
            )
            if result is not None:
                scored.append(result)

    if not scored:
        raise GraspPlanningFailure(f"No candidates passed IK scoring ({len(feasible)} tried)")

    scored.sort(key=lambda s: s.score, reverse=True)
    best = scored[0]
    logger.info(
        "Best grasp: %s yaw=%d tilt=%.0f° score=%.3f",
        best.grasp.face_label,
        best.grasp.yaw_variant,
        best.grasp.tilt_deg,
        best.score,
    )

    return best
