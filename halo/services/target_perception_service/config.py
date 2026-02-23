from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetPerceptionServiceConfig:
    fast_loop_hz: float = 10.0          # hint publish rate
    obs_age_limit_ms: int = 150         # gate: invalidate hint if obs is too old
    time_skew_limit_ms: int = 50        # gate: invalidate hint if camera/robot skew is too large
    reacquire_fail_limit: int = 3       # consecutive observe=None before REACQUIRE_FAILED
