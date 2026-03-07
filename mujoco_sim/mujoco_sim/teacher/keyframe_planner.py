"""SE(3) keyframe planner for scripted pick trajectories.

Generates a sequence of Cartesian keyframes (position + orientation + gripper)
from a scored GraspPose and home pose.  Each keyframe is tagged with a phase ID
and label so downstream modules can map trajectory segments to FSM phases.

The grasp orientation and approach direction come from ``GraspPose`` (produced
by ``grasp_planner.evaluate_grasps``).  Pregrasp is offset along the approach
direction; lift is always vertical.
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
    PHASE_VERIFY_GRASP,
)
from mujoco_sim.scene_info import TCP_PINCH_OFFSET_LOCAL
from mujoco_sim.teacher.grasp_planner import GraspPose


@dataclass
class Keyframe:
    """A single SE(3) waypoint in the pick trajectory."""

    position: np.ndarray  # (3,) world-frame XYZ
    orientation: np.ndarray  # (3,3) rotation matrix
    gripper: float  # GRIPPER_OPEN or GRIPPER_CLOSE
    phase_id: int  # FSM phase this keyframe represents
    label: str  # human-readable tag


def plan_pick_keyframes(
    home_joints: np.ndarray,
    grasp_pose: GraspPose,
    ee_site_id: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    standoff: float = 0.04,
    z_lift: float = 0.08,
) -> list[Keyframe]:
    """Plan Cartesian keyframes for a pick trajectory.

    Args:
        home_joints: (6,) home joint positions (arm + gripper).
        grasp_pose: GraspPose from the grasp planner (contact point, orientation,
            approach direction).
        ee_site_id: MuJoCo site id for gripperframe.
        model: MuJoCo model.
        data: MuJoCo data (used to forward-kinematics the home pose).
        standoff: Distance along approach direction for pregrasp offset.
        z_lift: Vertical lift height above grasp contact point.

    Returns:
        List of 6 Keyframe instances: home, pregrasp, grasp, grasp_closed, verify_grasp, lift.
    """
    # Compute home EE position via FK
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qpos[:6] = home_joints
    mujoco.mj_forward(model, d)
    home_pos = d.site_xpos[ee_site_id].copy()
    home_rot = d.site_xmat[ee_site_id].reshape(3, 3).copy()

    grasp_rot = grasp_pose.orientation

    # Contact positions (where we want the jaw midpoint to be)
    contact_grasp = grasp_pose.contact_point
    contact_pregrasp = contact_grasp - standoff * grasp_pose.approach_dir
    contact_lift = contact_grasp + np.array([0.0, 0.0, z_lift])

    # Shift so IK places the jaw midpoint (not gripperframe) at the contact point
    offset_world = grasp_rot @ TCP_PINCH_OFFSET_LOCAL
    pregrasp_pos = contact_pregrasp - offset_world
    grasp_pos = contact_grasp - offset_world
    lift_pos = contact_lift - offset_world

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
        Keyframe(
            position=lift_pos,
            orientation=grasp_rot,
            gripper=GRIPPER_CLOSE,
            phase_id=PHASE_VERIFY_GRASP,
            label="verify_grasp",
        ),
    ]
