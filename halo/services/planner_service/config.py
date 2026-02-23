from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlannerServiceConfig:
    watchdog_interval_s: float = 30.0  # tick even if no events arrive (safety net)
    max_commands_per_tick: int = 5
