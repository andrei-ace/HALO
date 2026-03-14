from __future__ import annotations

from dataclasses import dataclass

from halo.contracts.actions import SO101_DOF


@dataclass
class ControlServiceConfig:
    control_rate_hz: float = 50.0
    buffer_low_threshold_ms: int = 100  # publish BUFFER_LOW below this
    buffer_trim_ms: int = 75  # trim to ~this on phase switch

    # Joint-limit safety (SO-101 defaults)
    joint_limits_lower: tuple[float, ...] = (-1.92, -1.75, -1.69, -1.66, -2.74, -0.17)
    joint_limits_upper: tuple[float, ...] = (1.92, 1.75, 1.69, 1.66, 2.84, 1.75)
    max_joint_delta_rad: float = 0.5  # per-timestep velocity limit (optional, for TE-blended output)
    max_gripper_delta: float = 2.0  # full open/close span is 1.92 rad (-0.17..1.75)

    max_obs_age_ms: int = 200

    ensembling_temp: float = 0.01  # temporal ensembling decay (0.0 = uniform weight)

    def __post_init__(self) -> None:
        if len(self.joint_limits_lower) != SO101_DOF:
            raise ValueError(f"joint_limits_lower must have {SO101_DOF} elements, got {len(self.joint_limits_lower)}")
        if len(self.joint_limits_upper) != SO101_DOF:
            raise ValueError(f"joint_limits_upper must have {SO101_DOF} elements, got {len(self.joint_limits_upper)}")
