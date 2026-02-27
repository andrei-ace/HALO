"""DirectRLEnv for PICK — single-skill environment.

PICK-only environment. Success = cube lifted above height threshold and stable.

- Physics dt: 0.02s (50Hz), decimation: 5 -> 10Hz control rate
- Action space: 7 [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
- Observations: wrist_rgb, joint_pos, gripper_state, wrist_enabled
- Wrist camera gating: black image when phase not in WRIST_ACTIVE_PHASES
- IK: DifferentialIKController (damped least squares)

Note: This module requires Isaac Lab. It cannot be imported without Isaac Sim.
"""

from __future__ import annotations

import numpy as np

from halo_sim.cfg.env_cfg import PickEnvCfg
from halo_sim.constants import WRIST_ACTIVE_PHASES


class PickEnv:
    """PICK-only DirectRLEnv.

    This is a stub that documents the Isaac Lab environment interface.
    The actual implementation requires Isaac Sim/Lab runtime and will
    subclass isaaclab.envs.DirectRLEnv.

    When running inside Isaac Lab, replace this with:

        from isaaclab.envs import DirectRLEnv

        class PickEnv(DirectRLEnv):
            cfg: PickEnvCfg
            ...
    """

    def __init__(self, cfg: PickEnvCfg) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self._phase_ids: np.ndarray = np.zeros(cfg.num_envs, dtype=np.int32)
        self._lift_stable_count: np.ndarray = np.zeros(cfg.num_envs, dtype=np.int32)
        self._step_count: np.ndarray = np.zeros(cfg.num_envs, dtype=np.int32)

        # State arrays — populated by Isaac Lab in the real implementation;
        # initialised to sensible defaults for the stub.
        self._joint_pos: np.ndarray = np.zeros((cfg.num_envs, 7), dtype=np.float32)
        self._gripper_state: np.ndarray = np.zeros(cfg.num_envs, dtype=np.float32)
        self._ee_pos: np.ndarray = np.zeros((cfg.num_envs, 3), dtype=np.float32)
        self._ee_quat: np.ndarray = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (cfg.num_envs, 1))
        self._cube_pos: np.ndarray = np.zeros((cfg.num_envs, 3), dtype=np.float32)
        self._wrist_rgb: np.ndarray = np.zeros((cfg.num_envs, 240, 320, 3), dtype=np.uint8)

    def _setup_scene(self) -> None:
        """Called by Isaac Lab to set up the simulation scene.

        Creates:
        - Ground plane + dome light
        - Table (static rigid body)
        - Cube (dynamic rigid body, 4cm, 50g)
        - Robot (Franka Panda via FRANKA_PANDA_CFG)
        - Scene camera (TiledCameraCfg, 640x480, above table)
        - Wrist camera (TiledCameraCfg, 320x240, on EE link)
        - DifferentialIKController (damped least squares)
        """

    def _reset_idx(self, env_ids: np.ndarray) -> None:
        """Reset specified environments.

        Applies domain randomization:
        - Cube XY position: uniform jitter +-5cm
        - Lighting: uniform intensity variation
        - Cube mass: uniform range
        """
        self._step_count[env_ids] = 0
        self._lift_stable_count[env_ids] = 0
        self._phase_ids[env_ids] = 0

    def _pre_physics_step(self, actions: np.ndarray) -> None:
        """Convert EE-frame delta actions to joint targets via IK.

        actions: (num_envs, 7) [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]

        Steps:
        1. Clamp deltas to safety limits
        2. Apply IK (DifferentialIKController)
        3. Set joint position targets
        4. Apply gripper command
        """

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Build observation dict.

        Returns:
            wrist_rgb: (num_envs, H, W, 3) uint8 — black when wrist not active
            joint_pos: (num_envs, 7) float32
            gripper_state: (num_envs,) float32
            ee_pos: (num_envs, 3) float32
            ee_quat: (num_envs, 4) float32
            cube_pos: (num_envs, 3) float32
            wrist_enabled: (num_envs,) bool
        """
        wrist_enabled = np.array([p in WRIST_ACTIVE_PHASES for p in self._phase_ids])

        # Wrist camera gating: black image when phase not in WRIST_ACTIVE_PHASES
        wrist_rgb = self._wrist_rgb.copy()
        wrist_rgb[~wrist_enabled] = 0

        return {
            "wrist_rgb": wrist_rgb,
            "joint_pos": self._joint_pos,
            "gripper_state": self._gripper_state,
            "ee_pos": self._ee_pos,
            "ee_quat": self._ee_quat,
            "cube_pos": self._cube_pos,
            "wrist_enabled": wrist_enabled,
        }

    def _get_rewards(self) -> np.ndarray:
        """Compute per-env rewards (sparse: +1 on success, 0 otherwise)."""
        return np.zeros(self.num_envs, dtype=np.float32)

    def _get_dones(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute termination and truncation flags.

        Termination (success): cube above lift_height_threshold for lift_stability_steps.
        Truncation (timeout): step_count >= episode_timeout_steps.
        """
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = self._step_count >= self.cfg.episode_timeout_steps
        return terminated, truncated

    def set_phase_ids(self, phase_ids: np.ndarray) -> None:
        """Set current phase for wrist camera gating (called by teacher FSM)."""
        self._phase_ids = phase_ids
