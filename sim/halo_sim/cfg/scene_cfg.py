"""Scene configuration — PICK only: table + cube + cameras.

No bin (PICK only — lift is the terminal success).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CubeConfig:
    """Dynamic rigid body: 4cm cube, 50g."""

    prim_path: str = "{ENV_REGEX_NS}/Cube"
    size: float = 0.04  # meters
    mass: float = 0.05  # kg
    color: tuple[float, float, float] = (0.8, 0.2, 0.2)  # red
    # Spawn position (relative to env origin)
    spawn_pos: tuple[float, float, float] = (0.5, 0.0, 0.02)  # on table surface


@dataclass
class TableConfig:
    """Static rigid body: table."""

    prim_path: str = "{ENV_REGEX_NS}/Table"
    size: tuple[float, float, float] = (0.8, 1.2, 0.02)  # width, depth, thickness
    spawn_pos: tuple[float, float, float] = (0.5, 0.0, 0.0)  # table surface at z=0


@dataclass
class CameraConfig:
    """TiledCamera configuration."""

    prim_path: str = ""
    width: int = 640
    height: int = 480
    focal_length: float = 24.0


@dataclass
class SceneConfig:
    """Complete scene configuration for PICK environment."""

    cube: CubeConfig = field(default_factory=CubeConfig)
    table: TableConfig = field(default_factory=TableConfig)
    scene_camera: CameraConfig = field(
        default_factory=lambda: CameraConfig(
            prim_path="{ENV_REGEX_NS}/SceneCamera",
            width=640,
            height=480,
        )
    )
    wrist_camera: CameraConfig = field(
        default_factory=lambda: CameraConfig(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
            width=320,
            height=240,
        )
    )
    # Ground plane + dome light
    ground_plane: bool = True
    dome_light_intensity: float = 1000.0
