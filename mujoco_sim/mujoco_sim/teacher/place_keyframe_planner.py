"""SE(3) keyframe planner for scripted place trajectories.

Generates a sequence of Cartesian keyframes for placing a held object next to
a reference object. The arm moves to a preplace position above the target,
descends, opens the gripper, and retreats upward.

Phase mapping mirrors the PLACE FSM:
    TRANSIT_PREPLACE → DESCEND_PLACE → OPEN → RETREAT
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_sim.constants import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    PHASE_DESCEND_PLACE,
    PHASE_OPEN,
    PHASE_RETREAT,
    PHASE_TRANSIT_PREPLACE,
)
from mujoco_sim.teacher.keyframe_planner import Keyframe

# Default placement offset: place the held object next to the reference,
# offset along the X-axis by this amount (metres).
DEFAULT_PLACE_OFFSET_X = 0.04

# Height above the table surface for preplace hover
DEFAULT_PREPLACE_HEIGHT = 0.08

# Retreat height above the place point
DEFAULT_RETREAT_HEIGHT = 0.08


@dataclass
class PlaceConfig:
    """Configuration for place trajectory planning."""

    place_offset_x: float = DEFAULT_PLACE_OFFSET_X
    preplace_height: float = DEFAULT_PREPLACE_HEIGHT
    retreat_height: float = DEFAULT_RETREAT_HEIGHT


def plan_place_keyframes(
    current_joints: np.ndarray,
    reference_pos: np.ndarray,
    ee_site_id: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    table_z: float,
    held_half_sizes: np.ndarray,
    ref_half_sizes: np.ndarray,
    *,
    config: PlaceConfig | None = None,
) -> list[Keyframe]:
    """Plan Cartesian keyframes for a place trajectory.

    Args:
        current_joints: (6,) current joint positions (arm + gripper).
        reference_pos: (3,) world-frame position of the reference object.
        ee_site_id: MuJoCo site id for gripperframe.
        model: MuJoCo model.
        data: MuJoCo data.
        table_z: Table surface height (Z).
        held_half_sizes: (3,) half-sizes of the held object.
        ref_half_sizes: (3,) half-sizes of the reference object.
        config: Optional PlaceConfig overrides.

    Returns:
        List of 5 Keyframe instances: current, preplace, place, place_open, retreat.
    """
    cfg = config or PlaceConfig()

    # Compute current EE position via FK
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qpos[:6] = current_joints
    mujoco.mj_forward(model, d)
    current_pos = d.site_xpos[ee_site_id].copy()
    current_rot = d.site_xmat[ee_site_id].reshape(3, 3).copy()

    # Place target: next to the reference object on the table.
    # Offset along X by the sum of both objects' half-sizes + a gap.
    offset_x = ref_half_sizes[0] + held_half_sizes[0] + cfg.place_offset_x
    place_pos = reference_pos.copy()
    place_pos[0] += offset_x
    place_pos[2] = table_z + held_half_sizes[2]  # rest on table

    # Preplace: above the place position
    preplace_pos = place_pos.copy()
    preplace_pos[2] = place_pos[2] + cfg.preplace_height

    # Retreat: above the place position after releasing
    retreat_pos = place_pos.copy()
    retreat_pos[2] = place_pos[2] + cfg.retreat_height

    # Use a straight-down orientation (Z-axis = world -Z)
    place_rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    return [
        Keyframe(
            position=current_pos,
            orientation=current_rot,
            gripper=GRIPPER_CLOSE,
            phase_id=PHASE_TRANSIT_PREPLACE,
            label="current",
        ),
        Keyframe(
            position=preplace_pos,
            orientation=place_rot,
            gripper=GRIPPER_CLOSE,
            phase_id=PHASE_TRANSIT_PREPLACE,
            label="preplace",
        ),
        Keyframe(
            position=place_pos,
            orientation=place_rot,
            gripper=GRIPPER_CLOSE,
            phase_id=PHASE_DESCEND_PLACE,
            label="place",
        ),
        Keyframe(
            position=place_pos,
            orientation=place_rot,
            gripper=GRIPPER_OPEN,
            phase_id=PHASE_OPEN,
            label="place_open",
        ),
        Keyframe(
            position=retreat_pos,
            orientation=place_rot,
            gripper=GRIPPER_OPEN,
            phase_id=PHASE_RETREAT,
            label="retreat",
        ),
    ]
