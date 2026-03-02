"""Bridge configuration for HALO <-> MuJoCo sim ZMQ connection.

2-channel architecture:
    TelemetryStream (SUB): frames + state from sim
    CommandRPC (REQ): commands to sim (step, reset, start_pick, configure, set_hint)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass
class SimBridgeConfig:
    """Configuration for the HALO-side sim bridge client."""

    # Protocol version (v2 = telemetry + command RPC only)
    protocol_version: int = 2

    # ZMQ endpoints (HALO connects to these; sim server binds)
    telemetry_url: str = "tcp://127.0.0.1:5560"  # TelemetryStream: SUB (Sim → HALO)
    command_url: str = "tcp://127.0.0.1:5561"  # CommandRPC: REQ (HALO → Sim)
    hints_url: str | None = None  # Deprecated in protocol v2
    query_url: str | None = None  # Deprecated in protocol v2
    strict_mode: bool = False

    # Managed mode: HALO spawns the sim server subprocess
    managed: bool = False
    server_startup_timeout_s: float = 30.0

    # Timeouts
    recv_timeout_ms: int = 5000
    command_timeout_ms: int = 10_000
    heartbeat_timeout_ms: int = 5000

    # Frame settings
    wrist_rgb_height: int = 480
    wrist_rgb_width: int = 640

    def __post_init__(self) -> None:
        if self.protocol_version != 2:
            raise ValueError(f"Unsupported SimBridgeConfig.protocol_version={self.protocol_version}; expected 2")

        deprecated_fields = []
        if self.hints_url is not None:
            deprecated_fields.append("hints_url")
        if self.query_url is not None:
            deprecated_fields.append("query_url")

        if not deprecated_fields:
            return

        if self.strict_mode:
            fields = ", ".join(deprecated_fields)
            raise ValueError(f"{fields} are deprecated in protocol v2 (2-channel bridge)")

        warnings.warn(
            f"{', '.join(deprecated_fields)} are deprecated in protocol v2 and ignored",
            DeprecationWarning,
            stacklevel=2,
        )
