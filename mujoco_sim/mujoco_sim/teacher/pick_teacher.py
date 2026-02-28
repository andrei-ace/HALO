"""Scripted PICK teacher using privileged sim state.

Mirrors the PickFSM phase sequence from ``halo/services/skill_runner_service/fsm.py``
with the same distance thresholds from ``SkillRunnerConfig``.  Uses proportional
control in world-frame to generate EE-delta actions that approach, grasp, and lift
the target cube.

Usage::

    teacher = PickTeacher()
    teacher.reset()
    action, phase_id, done = teacher.step(obs)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mujoco_sim.constants import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    PHASE_CLOSE_GRIPPER,
    PHASE_DONE,
    PHASE_EXECUTE_APPROACH,
    PHASE_IDLE,
    PHASE_LIFT,
    PHASE_MOVE_PREGRASP,
    PHASE_PLAN_APPROACH,
    PHASE_SELECT_GRASP,
    PHASE_VERIFY_GRASP,
    PHASE_VISUAL_ALIGN,
)


@dataclass
class TeacherConfig:
    """Thresholds and gains for the scripted pick teacher.

    Distance thresholds mirror ``SkillRunnerConfig`` from HALO core.
    """

    # Phase transition distance thresholds (m) — same as SkillRunnerConfig
    approach_align_threshold_m: float = 0.15
    execute_approach_threshold_m: float = 0.05
    grasp_distance_threshold_m: float = 0.01

    # Timed phase durations (steps at control_freq)
    close_gripper_steps: int = 20  # 1 s at 20 Hz
    verify_steps: int = 10  # 0.5 s at 20 Hz
    lift_steps: int = 40  # 2 s at 20 Hz

    # Proportional control gains
    approach_gain: float = 0.8  # gain for MOVE_PREGRASP (coarse approach)
    align_gain: float = 0.5  # gain for VISUAL_ALIGN (slower, more precise)
    execute_gain: float = 0.3  # gain for EXECUTE_APPROACH (fine approach)
    lift_speed: float = 0.02  # m/step vertical lift increment

    # Pre-grasp offset: approach from above the cube
    pregrasp_height_offset: float = 0.05  # m above cube center

    # Action clamp (per-axis, m/step)
    max_delta: float = 0.05


class PickTeacher:
    """Scripted PICK policy using ground-truth sim state.

    Phase sequence mirrors PickFSM::

        IDLE → SELECT_GRASP → PLAN_APPROACH → MOVE_PREGRASP → VISUAL_ALIGN
        → EXECUTE_APPROACH → CLOSE_GRIPPER → VERIFY_GRASP → LIFT → DONE

    SELECT_GRASP and PLAN_APPROACH are immediate pass-throughs (same as v0 FSM).
    """

    def __init__(self, config: TeacherConfig | None = None) -> None:
        self._config = config or TeacherConfig()
        self._phase = PHASE_IDLE
        self._phase_step = 0

    def reset(self) -> None:
        """Reset teacher state for a new episode."""
        self._phase = PHASE_IDLE
        self._phase_step = 0

    @property
    def phase(self) -> int:
        """Current phase ID."""
        return self._phase

    @property
    def done(self) -> bool:
        """Whether the pick task is complete."""
        return self._phase == PHASE_DONE

    def step(self, obs: dict) -> tuple[np.ndarray, int, bool]:
        """Compute one teacher action from the current observation.

        Args:
            obs: Observation dict from ``RobosuiteEnv`` with keys:
                ``ee_pose`` (7,), ``object_pose`` (7,), ``gripper`` (float).

        Returns:
            (action, phase_id, done) where action is (7,) EE-delta.
        """
        ee_pos = obs["ee_pose"][:3]
        cube_pos = obs["object_pose"][:3]
        cfg = self._config

        # v0 pass-throughs: advance immediately
        if self._phase == PHASE_IDLE:
            self._transition(PHASE_SELECT_GRASP)
        if self._phase == PHASE_SELECT_GRASP:
            self._transition(PHASE_PLAN_APPROACH)
        if self._phase == PHASE_PLAN_APPROACH:
            self._transition(PHASE_MOVE_PREGRASP)

        distance = float(np.linalg.norm(cube_pos - ee_pos))

        action = np.zeros(7)

        if self._phase == PHASE_MOVE_PREGRASP:
            # Approach to pre-grasp position (above cube)
            target = cube_pos.copy()
            target[2] += cfg.pregrasp_height_offset
            delta = target - ee_pos
            action[:3] = np.clip(delta * cfg.approach_gain, -cfg.max_delta, cfg.max_delta)
            action[6] = GRIPPER_OPEN

            pregrasp_dist = float(np.linalg.norm(target - ee_pos))
            if pregrasp_dist < cfg.approach_align_threshold_m:
                self._transition(PHASE_VISUAL_ALIGN)

        elif self._phase == PHASE_VISUAL_ALIGN:
            # Descend toward cube, slower
            delta = cube_pos - ee_pos
            action[:3] = np.clip(delta * cfg.align_gain, -cfg.max_delta, cfg.max_delta)
            action[6] = GRIPPER_OPEN

            if distance < cfg.execute_approach_threshold_m:
                self._transition(PHASE_EXECUTE_APPROACH)

        elif self._phase == PHASE_EXECUTE_APPROACH:
            # Fine approach to grasp distance
            delta = cube_pos - ee_pos
            action[:3] = np.clip(delta * cfg.execute_gain, -cfg.max_delta, cfg.max_delta)
            action[6] = GRIPPER_OPEN

            if distance < cfg.grasp_distance_threshold_m:
                self._transition(PHASE_CLOSE_GRIPPER)

        elif self._phase == PHASE_CLOSE_GRIPPER:
            # Hold position, close gripper
            action[6] = GRIPPER_CLOSE
            self._phase_step += 1
            if self._phase_step >= cfg.close_gripper_steps:
                self._transition(PHASE_VERIFY_GRASP)

        elif self._phase == PHASE_VERIFY_GRASP:
            # Hold position, gripper closed, verify
            action[6] = GRIPPER_CLOSE
            self._phase_step += 1
            if self._phase_step >= cfg.verify_steps:
                self._transition(PHASE_LIFT)

        elif self._phase == PHASE_LIFT:
            # Lift straight up
            action[2] = cfg.lift_speed
            action[6] = GRIPPER_CLOSE
            self._phase_step += 1
            if self._phase_step >= cfg.lift_steps:
                self._transition(PHASE_DONE)

        return action, self._phase, self.done

    def _transition(self, new_phase: int) -> None:
        """Transition to a new phase, resetting the step counter."""
        self._phase = new_phase
        self._phase_step = 0
