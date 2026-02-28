"""Bridge configuration for HALO <-> MuJoCo sim ZMQ connection.

4-channel architecture:
    Ch1 (SUB): telemetry — frames + state from sim
    Ch2 (PUB): tracking hints to sim
    Ch3 (REQ): commands to sim (step, reset, teacher_step)
    Ch4 (REP): query service for sim (VLM detect, tracker init/update)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimBridgeConfig:
    """Configuration for the HALO-side sim bridge client."""

    # ZMQ endpoints (HALO connects to these; sim server binds)
    telemetry_url: str = "tcp://127.0.0.1:5560"  # Ch1: SUB (Sim → HALO)
    hints_url: str = "tcp://127.0.0.1:5561"  # Ch2: PUB (HALO → Sim)
    command_url: str = "tcp://127.0.0.1:5562"  # Ch3: REQ (HALO → Sim)
    query_url: str = "tcp://127.0.0.1:5563"  # Ch4: REP (Sim → HALO)

    # Managed mode: HALO spawns the sim server subprocess
    managed: bool = False
    server_startup_timeout_s: float = 30.0

    # Timeouts
    recv_timeout_ms: int = 5000
    command_timeout_ms: int = 10_000
    heartbeat_timeout_ms: int = 5000

    # Frame settings
    wrist_rgb_height: int = 240
    wrist_rgb_width: int = 320
