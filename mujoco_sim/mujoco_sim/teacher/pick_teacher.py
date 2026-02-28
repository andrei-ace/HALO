"""Scripted PICK teacher using privileged sim state.

Mirrors the PickFSM phase sequence from ``halo/services/skill_runner_service/fsm.py``
with the same distance thresholds from ``SkillRunnerConfig``.  Computes Cartesian
targets then solves IK → 6D joint-position actions for the SO-101.

Usage::

    teacher = PickTeacher()
    teacher.reset()
    action, phase_id, done = teacher.step(obs, model, data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
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
from mujoco_sim.teacher.ik_helper import solve_ik

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Thresholds and gains for the scripted pick teacher.

    Distance thresholds mirror ``SkillRunnerConfig`` from HALO core.
    """

    # Phase transition distance thresholds (m) — same as SkillRunnerConfig
    approach_align_threshold_m: float = 0.15
    execute_approach_threshold_m: float = 0.05
    grasp_distance_threshold_m: float = 0.03

    # Timed phase durations (steps at control_freq)
    close_gripper_steps: int = 20  # 1 s at 20 Hz
    verify_steps: int = 10  # 0.5 s at 20 Hz
    lift_steps: int = 120  # 6 s at 20 Hz

    # Pre-grasp offset: approach from above the cube
    pregrasp_height_offset: float = 0.08  # m above cube center
    grasp_height_offset: float = 0.02  # m above cube center — fingers straddle from above

    # Lift target height above cube initial position
    lift_height: float = 0.15  # m


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
        # Cache IDs on first step (need model)
        self._ee_site_id: int | None = None
        self._arm_joint_ids: list[int] | None = None

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

    def _ensure_ids(self, model: mujoco.MjModel) -> None:
        """Cache site/joint IDs on first call."""
        if self._ee_site_id is None:
            self._ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
            self._arm_joint_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
            ]

    def step(
        self,
        obs: dict,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[np.ndarray, int, bool]:
        """Compute one teacher action from the current observation.

        Args:
            obs: Observation dict from ``SO101Env`` with keys:
                ``ee_pose`` (7,), ``object_pose`` (7,), ``gripper`` (float), ``joint_pos`` (6,).
            model: MuJoCo model (for IK).
            data: MuJoCo data (for IK — will NOT be mutated).

        Returns:
            (action, phase_id, done) where action is (6,) joint-position targets.
        """
        self._ensure_ids(model)

        ee_pos = obs["ee_pose"][:3]
        cube_pos = obs["object_pose"][:3]
        current_joints = obs["joint_pos"]
        cfg = self._config

        # v0 pass-throughs: advance immediately
        if self._phase == PHASE_IDLE:
            self._transition(PHASE_SELECT_GRASP)
        if self._phase == PHASE_SELECT_GRASP:
            self._transition(PHASE_PLAN_APPROACH)
        if self._phase == PHASE_PLAN_APPROACH:
            self._transition(PHASE_MOVE_PREGRASP)

        distance = float(np.linalg.norm(cube_pos - ee_pos))

        # Default: hold current position
        arm_joints = current_joints[:5].copy()
        gripper_angle = float(current_joints[5])

        if self._phase == PHASE_MOVE_PREGRASP:
            # Approach to pre-grasp position (above cube)
            target = cube_pos.copy()
            target[2] += cfg.pregrasp_height_offset
            arm_joints = solve_ik(model, data, target, self._ee_site_id, self._arm_joint_ids)
            gripper_angle = GRIPPER_OPEN

            pregrasp_dist = float(np.linalg.norm(target - ee_pos))
            if pregrasp_dist < cfg.approach_align_threshold_m:
                self._transition(PHASE_VISUAL_ALIGN)

        elif self._phase == PHASE_VISUAL_ALIGN:
            # Descend toward cube
            arm_joints = solve_ik(model, data, cube_pos, self._ee_site_id, self._arm_joint_ids)
            gripper_angle = GRIPPER_OPEN

            if distance < cfg.execute_approach_threshold_m:
                self._transition(PHASE_EXECUTE_APPROACH)

        elif self._phase == PHASE_EXECUTE_APPROACH:
            # Fine approach — target at/below cube center so fingers straddle it
            target = cube_pos.copy()
            target[2] += cfg.grasp_height_offset
            arm_joints = solve_ik(model, data, target, self._ee_site_id, self._arm_joint_ids)
            gripper_angle = GRIPPER_OPEN

            target_dist = float(np.linalg.norm(target - ee_pos))
            if self._phase_step % 50 == 0:
                logger.debug(
                    "EXECUTE_APPROACH step=%d dist_cube=%.4f dist_target=%.4f ee=%s cube=%s",
                    self._phase_step,
                    distance,
                    target_dist,
                    ee_pos.round(4),
                    cube_pos.round(4),
                )

            if target_dist < cfg.grasp_distance_threshold_m:
                self._transition(PHASE_CLOSE_GRIPPER)

        elif self._phase == PHASE_CLOSE_GRIPPER:
            # Hold position, close gripper
            gripper_angle = GRIPPER_CLOSE
            self._phase_step += 1
            if self._phase_step >= cfg.close_gripper_steps:
                self._transition(PHASE_VERIFY_GRASP)

        elif self._phase == PHASE_VERIFY_GRASP:
            # Hold position, gripper closed, verify
            gripper_angle = GRIPPER_CLOSE
            self._phase_step += 1
            if self._phase_step >= cfg.verify_steps:
                self._transition(PHASE_LIFT)

        elif self._phase == PHASE_LIFT:
            # Lift: IK to a position above current EE
            lift_target = ee_pos.copy()
            lift_target[2] = cube_pos[2] + cfg.lift_height
            arm_joints = solve_ik(model, data, lift_target, self._ee_site_id, self._arm_joint_ids)
            gripper_angle = GRIPPER_CLOSE
            self._phase_step += 1
            if self._phase_step >= cfg.lift_steps:
                self._transition(PHASE_DONE)

        action = np.concatenate([arm_joints, [gripper_angle]])
        return action, self._phase, self.done

    def _transition(self, new_phase: int) -> None:
        """Transition to a new phase, resetting the step counter."""
        self._phase = new_phase
        self._phase_step = 0
