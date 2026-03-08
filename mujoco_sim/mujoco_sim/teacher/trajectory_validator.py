"""FK-based trajectory clearance validation.

Samples each trajectory segment at 5 points and checks that the gripper
stays above the table surface with sufficient margin.
"""

from __future__ import annotations

import logging

import mujoco

from mujoco_sim.teacher.trajectory import TrajectoryPlan
from mujoco_sim.teacher.waypoint_generator import JointWaypoint

logger = logging.getLogger(__name__)

GRIPPER_BODY_NAMES = ("gripper", "moving_jaw_so101_v1")


def get_gripper_collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    """Return collision geom IDs for the gripper body tree (cached by caller)."""
    geom_ids: list[int] = []
    for name in GRIPPER_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            continue
        start = model.body_geomadr[body_id]
        for i in range(model.body_geomnum[body_id]):
            gid = start + i
            if model.geom_contype[gid] > 0 or model.geom_conaffinity[gid] > 0:
                geom_ids.append(gid)
    return geom_ids


def validate_waypoint_gripper_clearance(
    waypoints: list[JointWaypoint],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_joint_ids: list[int],
    gripper_geom_ids: list[int],
    ee_site_id: int,
    table_z: float,
    margin: float = 0.01,
) -> bool:
    """Return True if all waypoints have gripper geometry above the table.

    For each waypoint, sets arm + gripper joint positions, runs FK, and checks
    that every gripper collision geom and the EE site are above ``table_z + margin``.

    Args:
        waypoints: Joint-space waypoints to check.
        model: MuJoCo model.
        data: MuJoCo data (used as template for qpos — not mutated).
        arm_joint_ids: List of 5 arm joint IDs.
        gripper_geom_ids: Collision geom IDs for gripper bodies.
        ee_site_id: MuJoCo site id for gripperframe.
        table_z: Table surface height (Z).
        margin: Minimum clearance above table (metres).

    Returns:
        True if all waypoints clear the threshold.
    """
    if not gripper_geom_ids:
        return True

    threshold = table_z + margin
    qpos_idx = [int(model.jnt_qposadr[jid]) for jid in arm_joint_ids]
    gripper_qpos_idx = int(model.jnt_qposadr[arm_joint_ids[-1]]) + 1  # gripper joint follows wrist_roll

    # Find the gripper joint qpos address
    gripper_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    if gripper_jnt_id != -1:
        gripper_qpos_idx = int(model.jnt_qposadr[gripper_jnt_id])

    d_fk = mujoco.MjData(model)

    for wp in waypoints:
        d_fk.qpos[:] = data.qpos[:]
        for i, _jid in enumerate(arm_joint_ids):
            d_fk.qpos[qpos_idx[i]] = wp.arm_joints[i]
        d_fk.qpos[gripper_qpos_idx] = wp.gripper
        mujoco.mj_forward(model, d_fk)

        # Check EE site
        ee_z = float(d_fk.site_xpos[ee_site_id][2])
        if ee_z < threshold:
            logger.info(
                "Gripper clearance violation at waypoint '%s': ee_z=%.4f < threshold=%.4f",
                wp.label,
                ee_z,
                threshold,
            )
            return False

        # Check each gripper collision geom
        for gid in gripper_geom_ids:
            geom_z = float(d_fk.geom_xpos[gid][2])
            if geom_z < threshold:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                logger.info(
                    "Gripper clearance violation at waypoint '%s': geom '%s' z=%.4f < threshold=%.4f",
                    wp.label,
                    geom_name,
                    geom_z,
                    threshold,
                )
                return False

    return True


def validate_trajectory_clearance(
    plan: TrajectoryPlan,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    arm_joint_ids: list[int],
    table_z: float,
    margin: float = 0.01,
) -> bool:
    """Return True if all sampled points have sufficient clearance above the table.

    For each segment, samples at t=0, t/4, t/2, 3t/4, t and runs FK to check
    that the EE Z position stays above ``table_z + margin``.

    Args:
        plan: Planned trajectory to validate.
        model: MuJoCo model.
        data: MuJoCo data (not mutated — used as template for qpos).
        ee_site_id: MuJoCo site id for gripperframe.
        arm_joint_ids: List of 5 arm joint IDs.
        table_z: Table surface height (Z).
        margin: Minimum clearance above table (metres).

    Returns:
        True if all sampled points clear the threshold.
    """
    threshold = table_z + margin
    qpos_idx = [int(model.jnt_qposadr[jid]) for jid in arm_joint_ids]
    d_fk = mujoco.MjData(model)

    for seg in plan.segments:
        if seg.duration < 1e-9:
            continue
        sample_times = [0.0, seg.duration * 0.25, seg.duration * 0.5, seg.duration * 0.75, seg.duration]
        for t in sample_times:
            arm_joints, _gripper = seg.sample(t)

            d_fk.qpos[:] = data.qpos[:]
            for i, jid in enumerate(arm_joint_ids):
                d_fk.qpos[qpos_idx[i]] = arm_joints[i]
            mujoco.mj_forward(model, d_fk)

            ee_z = float(d_fk.site_xpos[ee_site_id][2])
            if ee_z < threshold:
                logger.info(
                    "Clearance violation in segment '%s' at t=%.3f: ee_z=%.4f < threshold=%.4f",
                    seg.label,
                    t,
                    ee_z,
                    threshold,
                )
                return False

    return True
