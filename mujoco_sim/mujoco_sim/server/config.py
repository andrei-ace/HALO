"""Configuration for the MuJoCo sim server."""

from __future__ import annotations

from dataclasses import dataclass, field

from mujoco_sim.config import EnvConfig


@dataclass
class SimServerConfig:
    """Configuration for SimServer.

    Ports:
        telemetry_port (5560): TelemetryStream PUB — frames + state (Sim → HALO)
        command_port   (5561): CommandRPC REP — step/reset/start_pick/configure/set_hint (HALO → Sim)
    """

    host: str = "127.0.0.1"
    telemetry_port: int = 5560
    command_port: int = 5561

    # Rendering / publishing
    render_fps: int = 10
    jpeg_quality: int = 85

    # Physics loop rate (Hz) — independent of render_fps
    physics_hz: int = 20

    # Env config
    env_config: EnvConfig = field(default_factory=EnvConfig)

    @property
    def telemetry_url(self) -> str:
        return f"tcp://{self.host}:{self.telemetry_port}"

    @property
    def command_url(self) -> str:
        return f"tcp://{self.host}:{self.command_port}"
