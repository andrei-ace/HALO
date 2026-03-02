"""Raw MuJoCo environment wrapper for SO-101 with dual cameras and joint-position control."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim.config.env_config import EnvConfig

# Path to the MJCF assets directory
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "so101"


class SO101Env:
    """SO-101 environment wrapper for HALO PICK demos.

    Provides:
    - Dual cameras (scene + wrist) via offscreen rendering
    - 6-DOF joint-position action space [shoulder_pan, shoulder_lift, elbow_flex,
      wrist_flex, wrist_roll, gripper]
    - Seeded resets with cube position randomization
    - State get/set for VCR replay (qpos/qvel injection)
    """

    # Indices into qpos for the green cube freejoint (xyz component)
    _GREEN_CUBE_QPOS_X = 6  # after 6 robot joints
    _GREEN_CUBE_QPOS_Y = 7
    _GREEN_CUBE_QPOS_Z = 8

    # Indices into qpos for the red cube freejoint (after green cube's 7 DOFs)
    _RED_CUBE_QPOS_X = 13
    _RED_CUBE_QPOS_Y = 14
    _RED_CUBE_QPOS_Z = 15

    def __init__(self, config: EnvConfig | None = None) -> None:
        self._config = config or EnvConfig()

        xml_path = _ASSETS_DIR / self._config.scene_xml
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(
            self._model,
            height=max(
                self._config.scene_resolution[0],
                self._config.wrist_resolution[0],
            ),
            width=max(
                self._config.scene_resolution[1],
                self._config.wrist_resolution[1],
            ),
        )

        # Cache IDs
        self._scene_cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, self._config.scene_camera)
        self._ee_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._green_cube_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "green_cube")
        self._red_cube_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self._gripper_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")

        # Physics substeps: model.opt.timestep vs control_freq
        self._substeps = max(1, round(1.0 / (self._config.control_freq * self._model.opt.timestep)))

        self._step_count = 0

        # Store initial qpos as home pose (for stabilization)
        mujoco.mj_forward(self._model, self._data)
        self._home_qpos = self._data.qpos[:6].copy()

    def _extract_obs(self) -> dict[str, np.ndarray]:
        """Extract structured observation dict from MuJoCo state."""
        scene_h, scene_w = self._config.scene_resolution
        self._renderer.update_scene(self._data, camera=self._config.scene_camera)
        rgb_scene = self._renderer.render().copy()
        # Resize if renderer resolution doesn't match requested
        if rgb_scene.shape[0] != scene_h or rgb_scene.shape[1] != scene_w:
            rgb_scene = self._resize(rgb_scene, scene_h, scene_w)

        # Wrist camera (if present in MJCF)
        wrist_h, wrist_w = self._config.wrist_resolution
        try:
            self._renderer.update_scene(self._data, camera=self._config.wrist_camera)
            rgb_wrist = self._renderer.render().copy()
            if rgb_wrist.shape[0] != wrist_h or rgb_wrist.shape[1] != wrist_w:
                rgb_wrist = self._resize(rgb_wrist, wrist_h, wrist_w)
        except Exception:
            # Wrist camera not in MJCF yet — return black placeholder
            rgb_wrist = np.zeros((wrist_h, wrist_w, 3), dtype=np.uint8)

        # EE pose: site pos + body quaternion
        ee_pos = self._data.site_xpos[self._ee_site_id].copy()
        # Use rotation matrix → quaternion from the site
        ee_xmat = self._data.site_xmat[self._ee_site_id].reshape(3, 3)
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, ee_xmat.flatten())
        ee_pose = np.concatenate([ee_pos, ee_quat])

        # Cube pose (green)
        cube_pos = self._data.xpos[self._green_cube_body_id].copy()
        cube_quat = self._data.xquat[self._green_cube_body_id].copy()
        object_pose = np.concatenate([cube_pos, cube_quat])

        # Red cube pose
        red_cube_pos = self._data.xpos[self._red_cube_body_id].copy()
        red_cube_quat = self._data.xquat[self._red_cube_body_id].copy()
        red_object_pose = np.concatenate([red_cube_pos, red_cube_quat])

        # Joint positions for all 6 actuated joints (= action space state)
        joint_pos = self._data.qpos[:6].copy()

        return {
            "rgb_scene": rgb_scene,
            "rgb_wrist": rgb_wrist,
            "qpos": np.array(self._data.qpos, copy=True),
            "qvel": np.array(self._data.qvel, copy=True),
            "gripper": float(self._data.qpos[self._model.jnt_qposadr[self._gripper_joint_id]]),
            "ee_pose": ee_pose,
            "object_pose": object_pose,
            "red_object_pose": red_object_pose,
            "joint_pos": joint_pos,
        }

    @staticmethod
    def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
        """Nearest-neighbour resize without OpenCV dependency."""
        src_h, src_w = img.shape[:2]
        row_idx = (np.arange(h) * src_h / h).astype(int)
        col_idx = (np.arange(w) * src_w / w).astype(int)
        return img[np.ix_(row_idx, col_idx)]

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the environment, optionally with a seed for cube randomization.

        Returns dict with keys: rgb_scene, rgb_wrist, qpos, qvel, gripper,
        ee_pose, object_pose, joint_pos.
        """
        mujoco.mj_resetData(self._model, self._data)

        if seed is not None:
            rng = np.random.RandomState(seed)
            cx = rng.uniform(*self._config.green_cube_x_range)
            cy = rng.uniform(*self._config.green_cube_y_range)
            self._data.qpos[self._GREEN_CUBE_QPOS_X] = cx
            self._data.qpos[self._GREEN_CUBE_QPOS_Y] = cy
            # Z stays at default (from MJCF), quat stays identity

            # Randomize red cube with larger minimum separation from green cube.
            # This keeps detections/trackers from hopping between cubes.
            min_sep = 0.08
            for _ in range(100):
                rx = rng.uniform(*self._config.red_cube_x_range)
                ry = rng.uniform(*self._config.red_cube_y_range)
                if np.hypot(rx - cx, ry - cy) >= min_sep:
                    break
            self._data.qpos[self._RED_CUBE_QPOS_X] = rx
            self._data.qpos[self._RED_CUBE_QPOS_Y] = ry

        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        return self._extract_obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        """Step with a 6-DOF joint-position action.

        Args:
            action: (6,) array [shoulder_pan, shoulder_lift, elbow_flex,
                     wrist_flex, wrist_roll, gripper] — target joint positions.

        Returns:
            (obs, reward, done, info) tuple.
        """
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (6,), f"Expected action shape (6,), got {action.shape}"

        self._data.ctrl[:] = action
        for _ in range(self._substeps):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        done = self._step_count >= self._config.horizon
        obs = self._extract_obs()
        return obs, 0.0, done, {}

    def get_state(self) -> dict[str, np.ndarray]:
        """Get full MuJoCo state for replay/checkpointing."""
        return {
            "qpos": np.array(self._data.qpos, copy=True),
            "qvel": np.array(self._data.qvel, copy=True),
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        """Inject MuJoCo state for VCR replay."""
        self._data.qpos[:] = state["qpos"]
        self._data.qvel[:] = state["qvel"]
        mujoco.mj_forward(self._model, self._data)

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
        height = height or self._config.scene_resolution[0]
        width = width or self._config.scene_resolution[1]

        self._renderer.update_scene(self._data, camera=camera)
        img = self._renderer.render().copy()
        if img.shape[0] != height or img.shape[1] != width:
            img = self._resize(img, height, width)
        return img

    @property
    def action_dim(self) -> int:
        """Action dimensionality (6 for joint-position + gripper)."""
        return 6

    @property
    def mujoco_model(self) -> mujoco.MjModel:
        """Raw MuJoCo model."""
        return self._model

    @property
    def mujoco_data(self) -> mujoco.MjData:
        """Raw MuJoCo data."""
        return self._data

    @property
    def home_qpos(self) -> np.ndarray:
        """(6,) home joint positions from the MJCF default pose."""
        return self._home_qpos.copy()

    def close(self) -> None:
        """Close the environment and release resources."""
        self._renderer.close()
