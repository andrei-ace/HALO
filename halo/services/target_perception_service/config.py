from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetPerceptionServiceConfig:
    fast_loop_hz: float = 10.0  # hint publish rate
    obs_age_limit_ms: int = 150  # gate: invalidate hint if obs is too old
    time_skew_limit_ms: int = 50  # gate: invalidate hint if camera/robot skew is too large
    reacquire_fail_limit: int = 3  # consecutive observe=None before REACQUIRE_FAILED
    tracker_init_retries: int = 3  # VLM+replay attempts before declaring LOST
    frame_buffer_max_size: int = 300  # safety cap: max frames buffered during VLM inference
    capture_source_fps: float = 30.0  # expected FPS of capture source (for time-aware draining)
    max_frames_per_tick: int = 24  # hard cap on frames drained per tick
