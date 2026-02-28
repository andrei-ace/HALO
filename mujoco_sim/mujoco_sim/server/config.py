"""Configuration for the MuJoCo sim server."""

from __future__ import annotations

from dataclasses import dataclass, field

from mujoco_sim.config import EnvConfig
from mujoco_sim.teacher.pick_teacher import TeacherConfig


@dataclass
class SimServerConfig:
    """Configuration for SimServer.

    Ports:
        telemetry_port (5560): PUB — frames + state (Sim → HALO)
        hints_port     (5561): SUB — tracking hints (HALO → Sim)
        command_port   (5562): REP — step/reset/teacher_step (HALO → Sim)
        query_port     (5563): REQ — VLM/tracker queries (Sim → HALO)
    """

    host: str = "127.0.0.1"
    telemetry_port: int = 5560
    hints_port: int = 5561
    command_port: int = 5562
    query_port: int = 5563

    # Rendering / publishing
    render_fps: int = 10
    jpeg_quality: int = 85

    # Env + teacher
    env_config: EnvConfig = field(default_factory=EnvConfig)
    teacher_config: TeacherConfig = field(default_factory=TeacherConfig)

    # Timeouts
    query_timeout_ms: int = 30_000  # Ch4 REQ timeout for VLM/tracker queries

    @property
    def telemetry_url(self) -> str:
        return f"tcp://{self.host}:{self.telemetry_port}"

    @property
    def hints_url(self) -> str:
        return f"tcp://{self.host}:{self.hints_port}"

    @property
    def command_url(self) -> str:
        return f"tcp://{self.host}:{self.command_port}"

    @property
    def query_url(self) -> str:
        return f"tcp://{self.host}:{self.query_port}"
