"""Environment configuration for SO-101 + raw MuJoCo."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the SO101Env wrapper."""

    robot: str = "SO101"
    scene_xml: str = "pick_scene.xml"
    scene_camera: str = "scene_cam"
    wrist_camera: str = "wrist_cam"
    scene_resolution: tuple[int, int] = (480, 640)  # (H, W)
    wrist_resolution: tuple[int, int] = (240, 320)  # (H, W)
    control_freq: int = 20
    horizon: int = 1000
    cube_x_range: tuple[float, float] = (0.21, 0.27)  # reachable workspace in +X (1.5× further from base)
    cube_y_range: tuple[float, float] = (-0.06, 0.06)
