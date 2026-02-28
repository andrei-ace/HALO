"""Environment configuration for robosuite."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the RobosuiteEnv wrapper."""

    env_name: str = "Lift"
    robot: str = "Panda"
    controller: str = "BASIC"  # composite controller: OSC for arms, JOINT_POSITION for rest
    scene_camera: str = "agentview"
    wrist_camera: str = "robot0_eye_in_hand"
    scene_resolution: tuple[int, int] = (480, 640)  # (H, W)
    wrist_resolution: tuple[int, int] = (240, 320)  # (H, W)
    control_freq: int = 20
    horizon: int = 500
    has_renderer: bool = False
    has_offscreen_renderer: bool = True
