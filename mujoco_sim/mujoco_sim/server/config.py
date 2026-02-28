"""Configuration for the MuJoCo sim server."""

from __future__ import annotations

from dataclasses import dataclass, field

from mujoco_sim.config import EnvConfig
from mujoco_sim.teacher.pick_teacher import TeacherConfig


@dataclass
class SimServerConfig:
    """Configuration for SimServer.

    Ports:
        telemetry_port (5560): TelemetryStream PUB — frames + state (Sim → HALO)
        command_port   (5561): CommandRPC REP — step/reset/teacher_step/configure/set_hint (HALO → Sim)
    """

    host: str = "127.0.0.1"
    telemetry_port: int = 5560
    command_port: int = 5561

    # Rendering / publishing
    render_fps: int = 10
    jpeg_quality: int = 85

    # Env + teacher
    env_config: EnvConfig = field(default_factory=EnvConfig)
    teacher_config: TeacherConfig = field(default_factory=TeacherConfig)

    @property
    def telemetry_url(self) -> str:
        return f"tcp://{self.host}:{self.telemetry_port}"

    @property
    def command_url(self) -> str:
        return f"tcp://{self.host}:{self.command_port}"
