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


def _fk_current_ee(
    current_joints: np.ndarray,
    ee_site_id: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute current EE position and rotation via FK."""
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qpos[:6] = current_joints
    mujoco.mj_forward(model, d)
    return d.site_xpos[ee_site_id].copy(), d.site_xmat[ee_site_id].reshape(3, 3).copy()


# Number of candidate place positions to evaluate
DEFAULT_PLACE_N_CANDIDATES = 16

# Default placement offset: place the held object next to the reference,
# offset along the X-axis by this amount (metres).
DEFAULT_PLACE_OFFSET_X = 0.02

# Height above the table surface for preplace hover
DEFAULT_PREPLACE_HEIGHT = 0.08


@dataclass
class PlaceConfig:
    """Configuration for place trajectory planning."""

    place_offset_x: float = DEFAULT_PLACE_OFFSET_X
    preplace_height: float = DEFAULT_PREPLACE_HEIGHT


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
    current_pos, current_rot = _fk_current_ee(current_joints, ee_site_id, model, data)

    # Place target: next to the reference object on the table.
    # Offset along X by the sum of both objects' half-sizes + a gap.
    offset_x = ref_half_sizes[0] + held_half_sizes[0] + cfg.place_offset_x
    place_pos = reference_pos.copy()
    place_pos[0] += offset_x
    place_pos[2] = table_z + held_half_sizes[2]  # rest on table

    # Preplace: above the place position
    preplace_pos = place_pos.copy()
    preplace_pos[2] = place_pos[2] + cfg.preplace_height

    # Retreat: retrace to preplace position
    retreat_pos = preplace_pos.copy()

    return _build_place_keyframes(current_pos, current_rot, place_pos, preplace_pos, retreat_pos)


def _build_place_keyframes(
    current_pos: np.ndarray,
    current_rot: np.ndarray,
    place_pos: np.ndarray,
    preplace_pos: np.ndarray,
    retreat_pos: np.ndarray,
) -> list[Keyframe]:
    """Build the 5-keyframe place sequence from computed positions."""
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


def plan_place_candidates(
    current_joints: np.ndarray,
    reference_pos: np.ndarray,
    ee_site_id: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    table_z: float,
    held_half_sizes: np.ndarray,
    ref_half_sizes: np.ndarray,
    *,
    target_body: str = "",
    tray_floor_z: float | None = None,
    config: PlaceConfig | None = None,
    n_candidates: int = DEFAULT_PLACE_N_CANDIDATES,
) -> list[list[Keyframe]]:
    """Generate multiple candidate place keyframe sequences for trajectory validation.

    For a reference object: 16 positions in a circle around it at 22.5 degree increments.
    For floor placement (target_body == "floor"): 4x4 grid centered on gripper XY.

    Args:
        current_joints: (6,) current joint positions (arm + gripper).
        reference_pos: (3,) world-frame position of the reference object.
        ee_site_id: MuJoCo site id for gripperframe.
        model: MuJoCo model.
        data: MuJoCo data.
        table_z: Table surface height (Z).
        held_half_sizes: (3,) half-sizes of the held object.
        ref_half_sizes: (3,) half-sizes of the reference object.
        target_body: Name of the target body (use "floor" for floor placement,
            "tray" for tray placement).
        tray_floor_z: Z height of the tray interior floor (required when target_body == "tray").
        config: Optional PlaceConfig overrides.
        n_candidates: Number of candidate positions to generate.

    Returns:
        List of keyframe sequences (each is a list of 5 Keyframes).
    """
    cfg = config or PlaceConfig()
    current_pos, current_rot = _fk_current_ee(current_joints, ee_site_id, model, data)

    place_z = table_z + held_half_sizes[2]

    candidates: list[list[Keyframe]] = []

    if tray_floor_z is not None:
        # Single candidate centered on tray interior
        place_pos = np.array(
            [
                reference_pos[0],
                reference_pos[1],
                tray_floor_z + held_half_sizes[2],
            ]
        )
        preplace_pos = place_pos.copy()
        preplace_pos[2] = place_pos[2] + cfg.preplace_height
        retreat_pos = preplace_pos.copy()
        candidates.append(_build_place_keyframes(current_pos, current_rot, place_pos, preplace_pos, retreat_pos))
    elif target_body == "floor":
        # 4x4 grid centered on gripper's current XY
        grid_side = 4
        spacing = held_half_sizes[0] * 2 + 0.01
        for row in range(grid_side):
            for col in range(grid_side):
                offset_x = (col - (grid_side - 1) / 2) * spacing
                offset_y = (row - (grid_side - 1) / 2) * spacing
                place_pos = np.array(
                    [
                        current_pos[0] + offset_x,
                        current_pos[1] + offset_y,
                        place_z,
                    ]
                )
                preplace_pos = place_pos.copy()
                preplace_pos[2] = place_z + cfg.preplace_height
                retreat_pos = preplace_pos.copy()
                candidates.append(
                    _build_place_keyframes(current_pos, current_rot, place_pos, preplace_pos, retreat_pos)
                )
    else:
        # Circle around the reference object
        radius = ref_half_sizes[0] + held_half_sizes[0] + cfg.place_offset_x
        for i in range(n_candidates):
            angle = 2 * np.pi * i / n_candidates
            place_pos = np.array(
                [
                    reference_pos[0] + radius * np.cos(angle),
                    reference_pos[1] + radius * np.sin(angle),
                    place_z,
                ]
            )
            preplace_pos = place_pos.copy()
            preplace_pos[2] = place_z + cfg.preplace_height
            retreat_pos = preplace_pos.copy()
            candidates.append(_build_place_keyframes(current_pos, current_rot, place_pos, preplace_pos, retreat_pos))

    return candidates
