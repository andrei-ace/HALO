"""Jerk-limited trajectory planning using ruckig.

Converts joint-space waypoints into smooth, time-optimal trajectories with
per-joint velocity, acceleration, and jerk limits.  Each waypoint pair becomes
one ruckig segment; all segments start/end at rest (v=0, a=0).

Gripper is interpolated linearly within each segment (separate from ruckig).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ruckig import InputParameter, Ruckig, Trajectory

from mujoco_sim.teacher.waypoint_generator import JointWaypoint


@dataclass
class JointLimits:
    """Per-joint kinematic limits for trajectory generation."""

    max_velocity: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.5])
    max_acceleration: list[float] = field(default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 4.0])
    max_jerk: list[float] = field(default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 15.0])


@dataclass
class TrajectorySegment:
    """One segment of the pick trajectory (between two waypoints)."""

    trajectory: Trajectory  # ruckig Trajectory object
    duration: float  # seconds
    phase_id: int
    label: str
    gripper_start: float
    gripper_end: float

    def sample(self, t: float) -> tuple[np.ndarray, float]:
        """Sample arm joints (5,) and gripper at local time t.

        Args:
            t: Time within this segment [0, duration].

        Returns:
            (arm_joints, gripper) tuple.
        """
        t = np.clip(t, 0.0, self.duration)
        pos, _vel, _acc = self.trajectory.at_time(t)
        arm_joints = np.array(pos, dtype=np.float64)

        # Linear gripper interpolation
        if self.duration > 0:
            alpha = t / self.duration
        else:
            alpha = 1.0
        gripper = self.gripper_start + alpha * (self.gripper_end - self.gripper_start)

        return arm_joints, gripper


class TrajectoryPlan:
    """Full pick trajectory: sequence of segments with cumulative timing."""

    def __init__(self, segments: list[TrajectorySegment]) -> None:
        self.segments = segments
        self._cumulative_times: list[float] = []

        t = 0.0
        for seg in segments:
            self._cumulative_times.append(t)
            t += seg.duration
        self.total_duration = t

    def sample(self, t: float) -> tuple[np.ndarray, float, int]:
        """Sample (arm_joints, gripper, phase_id) at global time t.

        If t >= total_duration, returns the final waypoint.
        """
        t = np.clip(t, 0.0, self.total_duration)

        # Find which segment t falls into
        seg_idx = len(self.segments) - 1
        for i, cum_t in enumerate(self._cumulative_times):
            if i + 1 < len(self._cumulative_times) and t < self._cumulative_times[i + 1]:
                seg_idx = i
                break

        seg = self.segments[seg_idx]
        local_t = t - self._cumulative_times[seg_idx]
        arm_joints, gripper = seg.sample(local_t)
        return arm_joints, gripper, seg.phase_id


class TrajectoryError(Exception):
    """Raised when ruckig fails to compute a trajectory."""


def plan_trajectory(
    waypoints: list[JointWaypoint],
    joint_limits: JointLimits | None = None,
) -> TrajectoryPlan:
    """Plan a jerk-limited trajectory through joint-space waypoints.

    Each consecutive waypoint pair becomes one ruckig segment. All segments
    start and end at rest (v=0, a=0).

    Args:
        waypoints: Ordered joint-space waypoints (at least 2).
        joint_limits: Per-joint kinematic limits. Defaults to conservative values.

    Returns:
        TrajectoryPlan with smooth segments.

    Raises:
        TrajectoryError: If ruckig fails for any segment.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints for a trajectory")

    limits = joint_limits or JointLimits()
    n_dof = 5  # arm joints only

    segments: list[TrajectorySegment] = []

    for i in range(len(waypoints) - 1):
        wp_start = waypoints[i]
        wp_end = waypoints[i + 1]

        # Skip zero-motion segments (same position, only gripper changes)
        joint_delta = np.linalg.norm(wp_end.arm_joints - wp_start.arm_joints)
        if joint_delta < 1e-6:
            # Pure gripper segment — use a fixed duration for gripper close/open
            gripper_delta = abs(wp_end.gripper - wp_start.gripper)
            # ~0.5s for full gripper travel, scale proportionally
            duration = max(0.5 * gripper_delta / 1.5, 0.5)  # at least 500ms

            # Create a trivial ruckig trajectory (zero motion)
            otg = Ruckig(n_dof)
            inp = InputParameter(n_dof)
            traj = Trajectory(n_dof)

            inp.current_position = wp_start.arm_joints.tolist()
            inp.current_velocity = [0.0] * n_dof
            inp.current_acceleration = [0.0] * n_dof
            # Nudge target slightly to avoid ruckig zero-motion edge case
            target = wp_start.arm_joints.copy()
            target[0] += 1e-8
            inp.target_position = target.tolist()
            inp.target_velocity = [0.0] * n_dof
            inp.target_acceleration = [0.0] * n_dof
            inp.max_velocity = limits.max_velocity
            inp.max_acceleration = limits.max_acceleration
            inp.max_jerk = limits.max_jerk

            result = otg.calculate(inp, traj)
            if result < 0:
                raise TrajectoryError(
                    f"Ruckig failed for gripper segment '{wp_start.label}'→'{wp_end.label}': result={result}"
                )

            # Override with our desired duration if ruckig's is shorter
            actual_duration = max(traj.duration, duration)

            segments.append(
                TrajectorySegment(
                    trajectory=traj,
                    duration=actual_duration,
                    phase_id=wp_end.phase_id,
                    label=f"{wp_start.label}→{wp_end.label}",
                    gripper_start=wp_start.gripper,
                    gripper_end=wp_end.gripper,
                )
            )
            continue

        otg = Ruckig(n_dof)
        inp = InputParameter(n_dof)
        traj = Trajectory(n_dof)

        inp.current_position = wp_start.arm_joints.tolist()
        inp.current_velocity = [0.0] * n_dof
        inp.current_acceleration = [0.0] * n_dof
        inp.target_position = wp_end.arm_joints.tolist()
        inp.target_velocity = [0.0] * n_dof
        inp.target_acceleration = [0.0] * n_dof
        inp.max_velocity = limits.max_velocity
        inp.max_acceleration = limits.max_acceleration
        inp.max_jerk = limits.max_jerk

        result = otg.calculate(inp, traj)
        if result < 0:
            raise TrajectoryError(f"Ruckig failed for segment '{wp_start.label}'→'{wp_end.label}': result={result}")

        segments.append(
            TrajectorySegment(
                trajectory=traj,
                duration=traj.duration,
                phase_id=wp_end.phase_id,
                label=f"{wp_start.label}→{wp_end.label}",
                gripper_start=wp_start.gripper,
                gripper_end=wp_end.gripper,
            )
        )

    return TrajectoryPlan(segments)
