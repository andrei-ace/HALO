"""IK teacher — converts target EE pose to EE-frame delta actions.

Produces [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd] per step.
Clamps delta magnitude. Transforms world-frame delta to EE-frame.
"""

from __future__ import annotations

import numpy as np

from halo_sim.cfg.teacher_cfg import TeacherConfig
from halo_sim.constants import ACTION_DIM


class IKTeacher:
    """Converts target EE positions to EE-frame delta actions.

    For v0 milestone 1, orientation is held fixed (droll=dpitch=dyaw=0).
    Only positional deltas are generated.
    """

    def __init__(self, cfg: TeacherConfig) -> None:
        self._cfg = cfg

    def compute_actions(
        self,
        ee_pos: np.ndarray,  # (num_envs, 3) current EE position
        target_pos: np.ndarray,  # (num_envs, 3) target EE position
        gripper_cmd: np.ndarray,  # (num_envs,) gripper command
        ee_quat: np.ndarray | None = None,  # (num_envs, 4) current EE quaternion (unused in v0)
    ) -> np.ndarray:
        """Compute EE-frame delta actions.

        Returns:
            actions: (num_envs, 7) [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
        """
        num_envs = ee_pos.shape[0]
        actions = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)

        # Positional delta (world frame, v0: no EE-frame rotation)
        delta = target_pos - ee_pos

        # Clamp linear delta magnitude
        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        max_delta = self._cfg.max_linear_delta
        scale = np.where(delta_norm > max_delta, max_delta / (delta_norm + 1e-8), 1.0)
        delta = delta * scale

        # Fill action: [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
        actions[:, 0:3] = delta
        # v0: no orientation control (droll=dpitch=dyaw=0)
        actions[:, 6] = gripper_cmd

        return actions
