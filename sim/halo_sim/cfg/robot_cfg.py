"""Robot configuration — Franka Panda placeholder.

Uses Isaac Lab's built-in FRANKA_PANDA_CFG from isaaclab_assets.
Swap for SO-ARM101 URDF later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotConfig:
    """Franka Panda configuration for multi-env instancing."""

    # Prim path template — {ENV_REGEX_NS} is replaced per env instance
    prim_path: str = "{ENV_REGEX_NS}/Robot"
    # Isaac Lab asset config name
    asset_name: str = "FRANKA_PANDA_CFG"
    # End-effector link for IK
    ee_link_name: str = "panda_hand"
    # Gripper joint names
    gripper_joint_names: tuple[str, ...] = ("panda_finger_joint1", "panda_finger_joint2")
    # Arm joint names (7-DOF)
    arm_joint_names: tuple[str, ...] = (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )
    # Default joint positions (home config)
    default_joint_pos: tuple[float, ...] = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)
    # Gripper open/close widths (per finger)
    gripper_open_width: float = 0.04
    gripper_close_width: float = 0.0
