from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SkillRunnerConfig:
    runner_rate_hz: float = 10.0

    # Phase transition distance thresholds (m)
    approach_align_threshold_m: float = 0.15
    descend_threshold_m: float = 0.05
    grasp_distance_threshold_m: float = 0.01

    # GRASP_CLOSE deterministic trigger: hold below threshold for this long
    grasp_persistence_ms: int = 300

    # Per-phase timeouts
    approach_timeout_ms: int = 10_000
    align_timeout_ms: int = 5_000
    descend_timeout_ms: int = 5_000

    # Timed phases (no sensor exit condition in v0)
    close_duration_ms: int = 1_000
    verify_duration_ms: int = 500
    lift_duration_ms: int = 2_000

    # Recovery
    recover_wait_ms: int = 500
    max_reacquire_attempts: int = 3
    no_target_tolerance_ms: int = 2_000  # target absent this long → recovery

    # Chunk scheduling
    buffer_target_ms: int = 200
    chunk_horizon_steps: int = 10  # v0: 10 steps @ 10 Hz = 1 s horizon

    skip_verify_grasp: bool = False
