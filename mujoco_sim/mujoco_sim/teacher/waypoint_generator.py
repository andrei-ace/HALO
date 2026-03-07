"""Convert Cartesian keyframes → joint-space waypoints via orientation-aware IK.

Each keyframe is solved independently, seeded from the previous waypoint's joint
configuration to maintain continuity.  Falls back to yaw-rotated retries on IK
failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_sim.constants import (
    PHASE_DESCEND_PLACE,
    PHASE_LIFT,
    PHASE_OPEN,
    PHASE_RETREAT,
    PHASE_TRANSIT_PREPLACE,
)
from mujoco_sim.teacher.ik_helper import solve_ik, solve_ik_with_orientation
from mujoco_sim.teacher.keyframe_planner import Keyframe

logger = logging.getLogger(__name__)


@dataclass
class JointWaypoint:
    """A joint-space waypoint with metadata."""

    arm_joints: np.ndarray  # (5,) arm joint positions
    gripper: float  # gripper joint angle
    phase_id: int
    label: str


class IKFailure(Exception):
    """Raised when IK cannot reach a keyframe within tolerance."""


def generate_joint_waypoints(
    keyframes: list[Keyframe],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    arm_joint_ids: list[int],
    seed_joints: np.ndarray,
    *,
    pos_weight: float = 1.0,
    ori_weight: float = 0.1,
    max_iters: int = 200,
    tol: float = 2e-3,
    damping: float = 1e-2,
    pos_tol: float = 1e-2,
) -> list[JointWaypoint]:
    """Convert Cartesian keyframes to joint-space waypoints.

    Args:
        keyframes: Ordered SE(3) keyframes from the planner.
        model: MuJoCo model.
        data: MuJoCo data (will NOT be mutated).
        ee_site_id: MuJoCo site id for gripperframe.
        arm_joint_ids: List of 5 arm joint IDs.
        seed_joints: (5,) initial arm joint configuration.
        pos_weight: Position weight for IK.
        ori_weight: Orientation weight for IK.
        max_iters: Max IK iterations per keyframe.
        tol: IK convergence tolerance (position, metres).
        damping: IK damping coefficient.
        pos_tol: Maximum acceptable position error (metres). If exceeded, retries
            with rotated orientations.

    Returns:
        List of JointWaypoint, one per keyframe.

    Raises:
        IKFailure: If any keyframe cannot be reached within ``pos_tol``.
    """
    waypoints: list[JointWaypoint] = []

    # Working copy of data for FK checks
    d_check = mujoco.MjData(model)

    prev_joints = seed_joints.copy()
    all_solved: list[np.ndarray] = [seed_joints.copy()]

    for kf in keyframes:
        # Try seeding from the most recent solution first, then earlier ones
        seeds_to_try = [prev_joints] + list(reversed(all_solved[:-1]))
        solved = False

        # Lift and PLACE phases: 5-DOF arm can't fully control 6D pose,
        # orientation matters less for lifting/placing. Position-only IK with
        # relaxed tolerance (3 cm) avoids unnecessary IK failures.
        _POSITION_ONLY_PHASES = frozenset(
            {
                PHASE_LIFT,
                PHASE_TRANSIT_PREPLACE,
                PHASE_DESCEND_PLACE,
                PHASE_OPEN,
                PHASE_RETREAT,
            }
        )
        position_only = kf.phase_id in _POSITION_ONLY_PHASES
        kf_pos_tol = max(pos_tol, 0.03) if position_only else pos_tol

        for seed in seeds_to_try:
            try:
                joints = _solve_with_retries(
                    model=model,
                    data=data,
                    target_pos=kf.position,
                    target_rot=kf.orientation,
                    ee_site_id=ee_site_id,
                    arm_joint_ids=arm_joint_ids,
                    seed_joints=seed,
                    pos_weight=pos_weight,
                    ori_weight=ori_weight,
                    max_iters=max_iters,
                    tol=tol,
                    damping=damping,
                    pos_tol=kf_pos_tol,
                    d_check=d_check,
                    label=kf.label,
                    position_only=position_only,
                )
                solved = True
                break
            except IKFailure:
                continue

        if not solved:
            raise IKFailure(f"IK failed for keyframe '{kf.label}' after trying {len(seeds_to_try)} seed configurations")

        waypoints.append(
            JointWaypoint(
                arm_joints=joints,
                gripper=kf.gripper,
                phase_id=kf.phase_id,
                label=kf.label,
            )
        )
        prev_joints = joints.copy()
        all_solved.append(joints.copy())

    return waypoints


def _solve_with_retries(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    ee_site_id: int,
    arm_joint_ids: list[int],
    seed_joints: np.ndarray,
    pos_weight: float,
    ori_weight: float,
    max_iters: int,
    tol: float,
    damping: float,
    pos_tol: float,
    d_check: mujoco.MjData,
    label: str,
    position_only: bool = False,
) -> np.ndarray:
    """Solve IK with yaw-rotation fallbacks on failure."""
    # Map joint IDs → qpos indices (correct for any MJCF layout)
    qpos_idx = [int(model.jnt_qposadr[jid]) for jid in arm_joint_ids]

    # Try original orientation first
    yaw_offsets = [0.0, np.pi / 2, -np.pi / 2, np.pi]

    for yaw_offset in yaw_offsets:
        if yaw_offset != 0.0:
            # Rotate target orientation about world Z
            c, s = np.cos(yaw_offset), np.sin(yaw_offset)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            rot = Rz @ target_rot
        else:
            rot = target_rot

        # Seed the solver with previous joints
        d_seed = mujoco.MjData(model)
        d_seed.qpos[:] = data.qpos[:]
        for i, jid in enumerate(arm_joint_ids):
            d_seed.qpos[qpos_idx[i]] = seed_joints[i]
        mujoco.mj_forward(model, d_seed)

        if position_only:
            joints = solve_ik(
                model,
                d_seed,
                target_pos,
                ee_site_id,
                arm_joint_ids,
                max_iters=max_iters,
                tol=tol,
                damping=damping,
            )
        else:
            joints = solve_ik_with_orientation(
                model,
                d_seed,
                target_pos,
                rot,
                ee_site_id,
                arm_joint_ids,
                pos_weight=pos_weight,
                ori_weight=ori_weight,
                max_iters=max_iters,
                tol=tol,
                damping=damping,
            )

        # Check position error via FK
        d_check.qpos[:] = data.qpos[:]
        for i, jid in enumerate(arm_joint_ids):
            d_check.qpos[qpos_idx[i]] = joints[i]
        mujoco.mj_forward(model, d_check)
        ee_pos = d_check.site_xpos[ee_site_id]
        pos_err = float(np.linalg.norm(target_pos - ee_pos))

        if pos_err <= pos_tol:
            if yaw_offset != 0.0:
                logger.info(
                    "IK for '%s' succeeded with yaw offset %.1f° (err=%.4f m)",
                    label,
                    np.degrees(yaw_offset),
                    pos_err,
                )
            return joints

        logger.info(
            "IK for '%s' yaw_offset=%.1f° pos_err=%.4f m (tol=%.4f)",
            label,
            np.degrees(yaw_offset),
            pos_err,
            pos_tol,
        )

    raise IKFailure(
        f"IK failed for keyframe '{label}': position error {pos_err:.4f} m > {pos_tol:.4f} m after all retries"
    )
