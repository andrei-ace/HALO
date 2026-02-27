"""Environment configuration for PICK-only DirectRLEnv."""

from __future__ import annotations

from dataclasses import dataclass, field

from halo_sim.cfg.domain_rand import DomainRandConfig
from halo_sim.cfg.robot_cfg import RobotConfig
from halo_sim.cfg.scene_cfg import SceneConfig
from halo_sim.cfg.teacher_cfg import TeacherConfig


@dataclass
class PickEnvCfg:
    """Configuration for the PICK-only Isaac Lab environment.

    Physics dt: 0.02s (50Hz), decimation: 5 -> 10Hz control rate.
    """

    # Physics
    physics_dt: float = 0.02  # 50 Hz physics
    decimation: int = 5  # 50Hz / 5 = 10Hz control
    episode_timeout_steps: int = 300  # 30s at 10Hz

    # Action space: 7-DOF EE-frame deltas
    action_dim: int = 7

    # Success criteria
    lift_height_threshold: float = 0.08  # cube must be above this z to succeed
    lift_stability_steps: int = 5  # hold above threshold for N steps

    # IK controller
    ik_method: str = "dls"  # damped least squares

    # Multi-env
    num_envs: int = 64  # A6000 48GB default

    # Sub-configs
    robot: RobotConfig = field(default_factory=RobotConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
