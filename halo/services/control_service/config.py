from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlServiceConfig:
    control_rate_hz: float = 50.0
    buffer_low_threshold_ms: int = 100   # publish BUFFER_LOW below this
    buffer_trim_ms: int = 75             # trim to ~this on phase switch

    # SafetyGuard delta-magnitude limits
    max_linear_delta_m: float = 0.01     # per-timestep at 50 Hz ≈ 0.5 m/s
    max_angular_delta_rad: float = 0.02  # per-timestep at 50 Hz ≈ 1.0 rad/s
    max_gripper_delta: float = 1.0       # gripper can jump full range

    max_obs_age_ms: int = 200
