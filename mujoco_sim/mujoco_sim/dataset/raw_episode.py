"""In-memory episode buffer: Timestep dataclass + RawEpisode container."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Timestep:
    """Single simulation timestep.

    All arrays are stored as numpy.  Images are uint8 (H, W, 3).
    Poses are (7,) — [x, y, z, qx, qy, qz, qw].
    Action is (6,) — [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    (joint-position targets for SO-101).
    """

    rgb_scene: np.ndarray  # (H, W, 3) uint8
    rgb_wrist: np.ndarray  # (H, W, 3) uint8
    qpos: np.ndarray  # (nq,)
    qvel: np.ndarray  # (nv,)
    gripper: float
    ee_pose: np.ndarray  # (7,)
    action: np.ndarray  # (6,) joint-position targets
    phase_id: int | None = None  # teacher/detector phase label
    object_pose: np.ndarray | None = None  # (7,) optional
    joint_pos: np.ndarray | None = None  # (6,) all actuated joint positions, optional
    contacts: np.ndarray | None = None  # (N,) optional
    bbox_xywh: tuple[int, int, int, int] | None = None  # tracker bbox (x, y, w, h)
    tracker_ok: bool | None = None  # tracker status (True=tracking, False=lost)


@dataclass
class EpisodeMetadata:
    """Metadata attached to a recorded episode."""

    seed: int | None = None
    env_name: str = "PickScene"
    robot: str = "SO101"
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
    def joint_pos_array(self) -> np.ndarray | None:
        """(T, 6) or None if no timestep has joint_pos."""
        if not self._steps or self._steps[0].joint_pos is None:
            return None
        return np.stack([ts.joint_pos for ts in self._steps])

    @property
    def actions(self) -> np.ndarray:
        """(T, 6) joint-position targets."""
        return np.stack([ts.action for ts in self._steps])

    @property
    def phase_ids(self) -> np.ndarray | None:
        """(T,) int or None if no timestep has phase_id."""
        if not self._steps or self._steps[0].phase_id is None:
            return None
        return np.array([ts.phase_id for ts in self._steps], dtype=np.int32)

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

    @property
    def bbox_xywh_array(self) -> np.ndarray | None:
        """(T, 4) int32 or None if no timestep has bbox_xywh."""
        if not self._steps or self._steps[0].bbox_xywh is None:
            return None
        rows = [ts.bbox_xywh if ts.bbox_xywh is not None else (0, 0, 0, 0) for ts in self._steps]
        return np.array(rows, dtype=np.int32)

    @property
    def tracker_ok_array(self) -> np.ndarray | None:
        """(T,) bool or None if no timestep has tracker_ok."""
        if not self._steps or self._steps[0].tracker_ok is None:
            return None
        return np.array([ts.tracker_ok if ts.tracker_ok is not None else False for ts in self._steps], dtype=bool)
