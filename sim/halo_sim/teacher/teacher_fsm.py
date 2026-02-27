"""Teacher FSM — mirrors HALO PickFSM phases using privileged sim state.

Uses ground-truth cube pose to drive phase transitions. Produces
(phase_id, target_ee_pose, gripper_cmd) per step.
"""

from __future__ import annotations

import numpy as np

from halo_sim.cfg.teacher_cfg import TeacherConfig
from halo_sim.constants import (
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


class TeacherFSM:
    """Per-environment teacher FSM for batched demo generation.

    Operates on privileged sim state (ground-truth cube pose, EE pose).
    All operations are vectorized over num_envs.
    """

    def __init__(self, num_envs: int, cfg: TeacherConfig) -> None:
        self._cfg = cfg
        self.num_envs = num_envs
        self.phase_ids = np.full(num_envs, PHASE_IDLE, dtype=np.int32)
        self._phase_step_count = np.zeros(num_envs, dtype=np.int32)
        self._gripper_cmd = np.full(num_envs, GRIPPER_OPEN, dtype=np.float32)

    def reset(self, env_ids: np.ndarray) -> None:
        """Reset specified environments to start of PICK sequence."""
        self.phase_ids[env_ids] = PHASE_SELECT_GRASP
        self._phase_step_count[env_ids] = 0
        self._gripper_cmd[env_ids] = GRIPPER_OPEN

    def step(
        self,
        ee_pos: np.ndarray,  # (num_envs, 3)
        cube_pos: np.ndarray,  # (num_envs, 3)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute teacher actions based on privileged state.

        Returns:
            target_ee_pos: (num_envs, 3) target EE position
            gripper_cmd: (num_envs,) gripper command (0=open, 1=close)
            done: (num_envs,) bool — True when episode complete
        """
        cfg = self._cfg
        target_ee_pos = ee_pos.copy()
        done = np.zeros(self.num_envs, dtype=bool)

        for phase_id in [
            PHASE_SELECT_GRASP,
            PHASE_PLAN_APPROACH,
            PHASE_MOVE_PREGRASP,
            PHASE_VISUAL_ALIGN,
            PHASE_EXECUTE_APPROACH,
            PHASE_CLOSE_GRIPPER,
            PHASE_VERIFY_GRASP,
            PHASE_LIFT,
            PHASE_DONE,
        ]:
            mask = self.phase_ids == phase_id

            if phase_id == PHASE_SELECT_GRASP:
                # v0: immediate pass-through
                self.phase_ids[mask] = PHASE_PLAN_APPROACH
                self._phase_step_count[mask] = 0

            elif phase_id == PHASE_PLAN_APPROACH:
                # v0: immediate pass-through
                self.phase_ids[mask] = PHASE_MOVE_PREGRASP
                self._phase_step_count[mask] = 0

            elif phase_id == PHASE_MOVE_PREGRASP:
                # Move to pregrasp position (above cube)
                pregrasp = cube_pos.copy()
                pregrasp[:, 2] += cfg.pregrasp_offset_z
                target_ee_pos[mask] = pregrasp[mask]
                self._gripper_cmd[mask] = GRIPPER_OPEN

                # Check transition: close enough to pregrasp
                dist = np.linalg.norm(ee_pos[mask] - pregrasp[mask], axis=1)
                close_enough = dist < cfg.align_tolerance_xy * 3
                advance = mask.copy()
                advance[mask] = close_enough
                self.phase_ids[advance] = PHASE_VISUAL_ALIGN
                self._phase_step_count[advance] = 0

            elif phase_id == PHASE_VISUAL_ALIGN:
                # Fine alignment above cube
                align_target = cube_pos.copy()
                align_target[:, 2] += cfg.pregrasp_offset_z * 0.5
                target_ee_pos[mask] = align_target[mask]
                self._gripper_cmd[mask] = GRIPPER_OPEN

                dist_xy = np.linalg.norm(ee_pos[mask, :2] - cube_pos[mask, :2], axis=1)
                aligned = dist_xy < cfg.align_tolerance_xy
                advance = mask.copy()
                advance[mask] = aligned
                self.phase_ids[advance] = PHASE_EXECUTE_APPROACH
                self._phase_step_count[advance] = 0

            elif phase_id == PHASE_EXECUTE_APPROACH:
                # Descend to grasp
                grasp_target = cube_pos.copy()
                target_ee_pos[mask] = grasp_target[mask]
                self._gripper_cmd[mask] = GRIPPER_OPEN

                dist = np.linalg.norm(ee_pos[mask] - grasp_target[mask], axis=1)
                close_enough = dist < cfg.grasp_distance_threshold
                advance = mask.copy()
                advance[mask] = close_enough
                self.phase_ids[advance] = PHASE_CLOSE_GRIPPER
                self._phase_step_count[advance] = 0

            elif phase_id == PHASE_CLOSE_GRIPPER:
                target_ee_pos[mask] = ee_pos[mask]  # hold position
                self._gripper_cmd[mask] = GRIPPER_CLOSE
                self._phase_step_count[mask] += 1

                timer_done = self._phase_step_count[mask] >= cfg.gripper_close_steps
                advance = mask.copy()
                advance[mask] = timer_done
                self.phase_ids[advance] = PHASE_VERIFY_GRASP
                self._phase_step_count[advance] = 0

            elif phase_id == PHASE_VERIFY_GRASP:
                target_ee_pos[mask] = ee_pos[mask]  # hold position
                self._gripper_cmd[mask] = GRIPPER_CLOSE
                self._phase_step_count[mask] += 1

                timer_done = self._phase_step_count[mask] >= cfg.verify_steps
                advance = mask.copy()
                advance[mask] = timer_done
                self.phase_ids[advance] = PHASE_LIFT
                self._phase_step_count[advance] = 0

            elif phase_id == PHASE_LIFT:
                lift_target = ee_pos.copy()
                lift_target[:, 2] = cfg.lift_height
                target_ee_pos[mask] = lift_target[mask]
                self._gripper_cmd[mask] = GRIPPER_CLOSE

                at_height = ee_pos[mask, 2] >= cfg.lift_height - 0.01
                advance = mask.copy()
                advance[mask] = at_height
                self.phase_ids[advance] = PHASE_DONE
                self._phase_step_count[advance] = 0

            elif phase_id == PHASE_DONE:
                target_ee_pos[mask] = ee_pos[mask]
                self._gripper_cmd[mask] = GRIPPER_CLOSE
                done[mask] = True

        return target_ee_pos, self._gripper_cmd.copy(), done
