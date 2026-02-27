"""Teacher configuration — tolerances and speeds for IK teacher."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TeacherConfig:
    """Configuration for the analytic IK teacher."""

    # Approach
    pregrasp_offset_z: float = 0.10  # hover above cube (m)
    approach_speed: float = 0.02  # max delta per step (m)

    # Alignment
    align_tolerance_xy: float = 0.005  # alignment precision (m)
    align_speed: float = 0.01  # max delta per step (m)

    # Execute approach (descend)
    descend_speed: float = 0.005  # max delta per step (m)
    grasp_distance_threshold: float = 0.005  # when to close gripper (m)

    # Gripper
    gripper_close_steps: int = 10  # steps to hold gripper closed
    gripper_force: float = 40.0  # N (Franka gripper max ~70N)

    # Lift
    lift_height: float = 0.12  # target lift height above table (m)
    lift_speed: float = 0.01  # max delta per step (m)

    # Verify
    verify_steps: int = 5  # steps to verify grasp stability

    # Max delta clamp (safety, per-step)
    max_linear_delta: float = 0.03  # m
    max_angular_delta: float = 0.1  # rad
