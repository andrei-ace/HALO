"""Bridge configuration for HALO <-> Isaac Lab sim connection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimBridgeConfig:
    sim_action_url: str = "tcp://localhost:5555"
    sim_obs_url: str = "tcp://localhost:5556"
    recv_timeout_ms: int = 5000
    wrist_rgb_height: int = 240
    wrist_rgb_width: int = 320
