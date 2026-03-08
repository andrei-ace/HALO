"""Environment configuration for SO-101 + raw MuJoCo."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the SO101Env wrapper."""

    robot: str = "SO101"
    scene_xml: str = "pick_scene.xml"
    scene_camera: str = "scene_cam"
    wrist_camera: str = "wrist_cam"
    scene_resolution: tuple[int, int] = (720, 1280)  # (H, W)
    wrist_resolution: tuple[int, int] = (480, 640)  # (H, W)
    control_freq: int = 20
    horizon: int = 1_000_000
    # Reachable workspace on the table (table spans X:-0.28..0.52, Y:-0.35..0.35)
    # Keep objects within arm reach (~0.35m) and away from table edges
    green_cube_x_range: tuple[float, float] = (0.18, 0.32)
    green_cube_y_range: tuple[float, float] = (-0.15, 0.15)
    red_cube_x_range: tuple[float, float] = (0.18, 0.32)
    red_cube_y_range: tuple[float, float] = (-0.15, 0.15)
    tray_x_range: tuple[float, float] = (0.22, 0.40)
    tray_y_range: tuple[float, float] = (-0.15, 0.15)
