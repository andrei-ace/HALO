"""SE(3) keyframe planner for scripted pick trajectories.

Generates a sequence of Cartesian keyframes (position + orientation + gripper)
from cube pose and home pose.  Each keyframe is tagged with a phase ID and label
so downstream modules can map trajectory segments to FSM phases.

Orientation rule: "gripper down" = gripperframe Z-axis aligned with world -Z.
Yaw is extracted from the cube quaternion (rotation about world Z).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_sim.constants import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    PHASE_CLOSE_GRIPPER,
    PHASE_EXECUTE_APPROACH,
    PHASE_IDLE,
    PHASE_LIFT,
    PHASE_MOVE_PREGRASP,
)


@dataclass
class Keyframe:
    """A single SE(3) waypoint in the pick trajectory."""

    position: np.ndarray  # (3,) world-frame XYZ
    orientation: np.ndarray  # (3,3) rotation matrix
    gripper: float  # GRIPPER_OPEN or GRIPPER_CLOSE
    phase_id: int  # FSM phase this keyframe represents
    label: str  # human-readable tag


def _gripper_down_rotation(yaw: float = 0.0) -> np.ndarray:
    """Build a rotation matrix with Z-axis pointing down (world -Z) and given yaw.

    The resulting frame has:
    - Z-column = [0, 0, -1]  (gripper pointing down)
    - X-column = [cos(yaw), sin(yaw), 0]
    - Y-column = [-sin(yaw), cos(yaw), 0]  (right-hand rule adjusted for Z down)

    Actually, for Z = [0,0,-1] and X = [cos(yaw), sin(yaw), 0]:
    Y = Z × X (right-hand rule to ensure det(R)=+1)
    """
    c, s = np.cos(yaw), np.sin(yaw)
    x_axis = np.array([c, s, 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R


def _yaw_from_quat(quat: np.ndarray) -> float:
    """Extract yaw (rotation about world Z) from a quaternion [w, x, y, z]."""
    w, x, y, z = quat
    # atan2(2(wz + xy), 1 - 2(y² + z²))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def plan_pick_keyframes(
    home_joints: np.ndarray,
    cube_pos: np.ndarray,
    cube_quat: np.ndarray,
    ee_site_id: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    z_hover: float = 0.08,
    z_grasp: float = 0.0,
    z_lift: float = 0.08,
) -> list[Keyframe]:
    """Plan Cartesian keyframes for a pick trajectory.

    Args:
        home_joints: (6,) home joint positions (arm + gripper).
        cube_pos: (3,) cube center in world frame.
        cube_quat: (4,) cube orientation quaternion [w,x,y,z].
        ee_site_id: MuJoCo site id for gripperframe.
        model: MuJoCo model.
        data: MuJoCo data (used to forward-kinematics the home pose).
        z_hover: Height above cube for pregrasp.
        z_grasp: Height offset from cube center for grasp.
        z_lift: Height above cube for lift.

    Returns:
        List of 5 Keyframe instances: home, pregrasp, grasp, grasp_closed, lift.
    """
    # Compute home EE position via FK
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qpos[:6] = home_joints
    mujoco.mj_forward(model, d)
    home_pos = d.site_xpos[ee_site_id].copy()
    home_rot = d.site_xmat[ee_site_id].reshape(3, 3).copy()

    # Orientation: gripper-down with yaw from cube
    yaw = _yaw_from_quat(cube_quat)
    grasp_rot = _gripper_down_rotation(yaw)

    # Positions
    pregrasp_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + z_hover])
    grasp_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + z_grasp])
    lift_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + z_lift])

    return [
        Keyframe(
            position=home_pos,
            orientation=home_rot,
            gripper=GRIPPER_OPEN,
            phase_id=PHASE_IDLE,
            label="home",
        ),
        Keyframe(
            position=pregrasp_pos,
            orientation=grasp_rot,
            gripper=GRIPPER_OPEN,
            phase_id=PHASE_MOVE_PREGRASP,
            label="pregrasp",
        ),
        Keyframe(
            position=grasp_pos,
            orientation=grasp_rot,
            gripper=GRIPPER_OPEN,
            phase_id=PHASE_EXECUTE_APPROACH,
            label="grasp",
        ),
        Keyframe(
            position=grasp_pos,
            orientation=grasp_rot,
            gripper=GRIPPER_CLOSE,
            phase_id=PHASE_CLOSE_GRIPPER,
            label="grasp_closed",
        ),
        Keyframe(
            position=lift_pos,
            orientation=grasp_rot,
            gripper=GRIPPER_CLOSE,
            phase_id=PHASE_LIFT,
            label="lift",
        ),
    ]
