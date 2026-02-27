"""Domain randomization configuration — V0 mild."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainRandConfig:
    """V0 mild randomization applied in _reset_idx."""

    # Cube XY position jitter (uniform, meters)
    cube_xy_jitter: float = 0.05  # +-5cm

    # Lighting variation
    light_intensity_range: tuple[float, float] = (800.0, 1200.0)

    # Cube mass variation (uniform, kg)
    cube_mass_range: tuple[float, float] = (0.03, 0.08)

    # Table friction variation
    table_friction_range: tuple[float, float] = (0.4, 0.8)
