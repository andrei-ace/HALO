"""Thin wrapper around robosuite.make() with dual cameras and EE-delta action space."""

from __future__ import annotations

import sys
import types

import numpy as np

from mujoco_sim.config.env_config import EnvConfig


def _suppress_robosuite_warnings() -> None:
    """Inject a fake ``macros_private`` module so robosuite skips its startup warnings.

    Must be called **before** ``import robosuite``.  Suppresses: missing
    macros_private, optional robosuite_models/mink, and controller-component
    mismatches for Panda (all harmless for our single-arm Lift env).
    """
    if "robosuite.macros_private" not in sys.modules:
        fake = types.ModuleType("robosuite.macros_private")
        fake.CONSOLE_LOGGING_LEVEL = "ERROR"  # type: ignore[attr-defined]
        sys.modules["robosuite.macros_private"] = fake


class RobosuiteEnv:
    """Robosuite environment wrapper for HALO PICK demos.

    Provides:
    - Dual cameras (scene + wrist) via offscreen rendering
    - 7-DOF EE-delta action space [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
    - Seeded resets for reproducibility
    - State get/set for VCR replay (qpos/qvel injection)
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        _suppress_robosuite_warnings()
        import robosuite as suite

        self._config = config or EnvConfig()

        controller_config = suite.load_composite_controller_config(controller="BASIC")
        self._env = suite.make(
            env_name=self._config.env_name,
            robots=self._config.robot,
            controller_configs=controller_config,
            has_renderer=self._config.has_renderer,
            has_offscreen_renderer=self._config.has_offscreen_renderer,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=[self._config.scene_camera, self._config.wrist_camera],
            camera_heights=[self._config.scene_resolution[0], self._config.wrist_resolution[0]],
            camera_widths=[self._config.scene_resolution[1], self._config.wrist_resolution[1]],
            control_freq=self._config.control_freq,
            horizon=self._config.horizon,
        )

    def _extract_obs(self, raw_obs: dict) -> dict[str, np.ndarray]:
        """Extract structured observation dict from raw robosuite obs."""
        scene_key = f"{self._config.scene_camera}_image"
        wrist_key = f"{self._config.wrist_camera}_image"

        ee_pos = raw_obs["robot0_eef_pos"]  # (3,)
        ee_quat = raw_obs["robot0_eef_quat"]  # (4,)
        ee_pose = np.concatenate([ee_pos, ee_quat])  # (7,)

        cube_pos = raw_obs.get("cube_pos", np.zeros(3))
        cube_quat = raw_obs.get("cube_quat", np.array([1.0, 0.0, 0.0, 0.0]))
        object_pose = np.concatenate([cube_pos, cube_quat])  # (7,)

        gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
        gripper = float(gripper_qpos[0]) if gripper_qpos.size > 0 else 0.0

        return {
            "rgb_scene": raw_obs[scene_key],  # (H, W, 3)
            "rgb_wrist": raw_obs[wrist_key],  # (H, W, 3)
            "qpos": np.array(self._env.sim.data.qpos, copy=True),
            "qvel": np.array(self._env.sim.data.qvel, copy=True),
            "gripper": gripper,
            "ee_pose": ee_pose,  # (7,) pos + quat
            "object_pose": object_pose,  # (7,) pos + quat
        }

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the environment, optionally with a seed for reproducibility.

        Returns dict with keys: rgb_scene, rgb_wrist, qpos, qvel, gripper, ee_pose, object_pose.
        """
        if seed is not None:
            np.random.seed(seed)
        raw_obs = self._env.reset()
        return self._extract_obs(raw_obs)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        """Step the environment with a 7-DOF EE-delta action.

        Args:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]

        Returns:
            (obs, reward, done, info) tuple.
        """
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (7,), f"Expected action shape (7,), got {action.shape}"
        raw_obs, reward, done, info = self._env.step(action)
        return self._extract_obs(raw_obs), float(reward), bool(done), info

    def get_state(self) -> dict[str, np.ndarray]:
        """Get full MuJoCo state for replay/checkpointing."""
        return {
            "qpos": np.array(self._env.sim.data.qpos, copy=True),
            "qvel": np.array(self._env.sim.data.qvel, copy=True),
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        """Inject MuJoCo state for VCR replay.

        Sets qpos/qvel and calls mj_forward to recompute derived quantities.
        """
        import mujoco

        sim = self._env.sim
        sim.data.qpos[:] = state["qpos"]
        sim.data.qvel[:] = state["qvel"]
        mujoco.mj_forward(sim.model._model, sim.data._data)

    def render(self, camera: str | None = None, width: int | None = None, height: int | None = None) -> np.ndarray:
        """Offscreen render from a named camera.

        Args:
            camera: Camera name. Defaults to scene camera.
            width: Image width. Defaults to scene resolution width.
            height: Image height. Defaults to scene resolution height.

        Returns:
            RGB image as (H, W, 3) uint8 array.
        """
        camera = camera or self._config.scene_camera
        width = width or self._config.scene_resolution[1]
        height = height or self._config.scene_resolution[0]
        return self._env.sim.render(camera_name=camera, width=width, height=height)[::-1]

    @property
    def action_dim(self) -> int:
        """Action dimensionality (always 7 for EE-delta + gripper)."""
        return 7

    @property
    def mujoco_model(self):
        """Raw MuJoCo model (mjModel)."""
        return self._env.sim.model._model

    @property
    def mujoco_data(self):
        """Raw MuJoCo data (mjData)."""
        return self._env.sim.data._data

    @property
    def unwrapped(self):
        """Access the underlying robosuite environment."""
        return self._env

    def close(self) -> None:
        """Close the environment and release resources."""
        self._env.close()
