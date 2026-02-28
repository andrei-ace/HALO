"""In-memory episode buffer: Timestep dataclass + RawEpisode container."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Timestep:
    """Single simulation timestep.

    All arrays are stored as numpy.  Images are uint8 (H, W, 3).
    Poses are (7,) — [x, y, z, qx, qy, qz, qw].
    Action is (7,) — [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd].
    """

    rgb_scene: np.ndarray  # (H, W, 3) uint8
    rgb_wrist: np.ndarray  # (H, W, 3) uint8
    qpos: np.ndarray  # (nq,)
    qvel: np.ndarray  # (nv,)
    gripper: float
    ee_pose: np.ndarray  # (7,)
    action: np.ndarray  # (7,)
    object_pose: np.ndarray | None = None  # (7,) optional
    contacts: np.ndarray | None = None  # (N,) optional


@dataclass
class EpisodeMetadata:
    """Metadata attached to a recorded episode."""

    seed: int | None = None
    env_name: str = "Lift"
    robot: str = "Panda"
    control_freq: int = 20
    extra: dict = field(default_factory=dict)


class RawEpisode:
    """In-memory buffer of timesteps for one episode.

    Supports append, len, indexing, slicing, and bulk numpy accessors.
    """

    def __init__(self, metadata: EpisodeMetadata | None = None) -> None:
        self._steps: list[Timestep] = []
        self.metadata = metadata or EpisodeMetadata()

    def append(self, ts: Timestep) -> None:
        """Append a timestep to the episode."""
        self._steps.append(ts)

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, idx: int | slice) -> Timestep | list[Timestep]:
        return self._steps[idx]

    # ------------------------------------------------------------------
    # Bulk numpy accessors — return (T, ...) arrays over all timesteps
    # ------------------------------------------------------------------

    @property
    def rgb_scenes(self) -> np.ndarray:
        """(T, H, W, 3) uint8."""
        return np.stack([ts.rgb_scene for ts in self._steps])

    @property
    def rgb_wrists(self) -> np.ndarray:
        """(T, H, W, 3) uint8."""
        return np.stack([ts.rgb_wrist for ts in self._steps])

    @property
    def qpos_array(self) -> np.ndarray:
        """(T, nq)."""
        return np.stack([ts.qpos for ts in self._steps])

    @property
    def qvel_array(self) -> np.ndarray:
        """(T, nv)."""
        return np.stack([ts.qvel for ts in self._steps])

    @property
    def gripper_array(self) -> np.ndarray:
        """(T,)."""
        return np.array([ts.gripper for ts in self._steps])

    @property
    def ee_poses(self) -> np.ndarray:
        """(T, 7)."""
        return np.stack([ts.ee_pose for ts in self._steps])

    @property
    def actions(self) -> np.ndarray:
        """(T, 7)."""
        return np.stack([ts.action for ts in self._steps])

    @property
    def object_poses(self) -> np.ndarray | None:
        """(T, 7) or None if no timestep has object_pose."""
        if not self._steps or self._steps[0].object_pose is None:
            return None
        return np.stack([ts.object_pose for ts in self._steps])

    @property
    def contacts_list(self) -> list[np.ndarray | None]:
        """List of per-timestep contact arrays (variable length)."""
        return [ts.contacts for ts in self._steps]
