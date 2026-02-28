"""HDF5 episode writer — one file per episode with gzip-compressed images."""

from __future__ import annotations

import datetime
from pathlib import Path

import h5py

from mujoco_sim.dataset.raw_episode import RawEpisode


def write_episode(episode: RawEpisode, path: str | Path) -> Path:
    """Write a RawEpisode to an HDF5 file.

    Layout::

        /obs/rgb_scene   (T, H, W, 3) uint8  — gzip
        /obs/rgb_wrist   (T, H, W, 3) uint8  — gzip
        /obs/qpos        (T, nq)
        /obs/qvel        (T, nv)
        /obs/gripper     (T,)
        /obs/ee_pose     (T, 7)
        /obs/object_pose (T, 7)            — only if present
        /obs/contacts/step_NNNNNN (N,)     — only if present
        /action          (T, 7)
        attrs: seed, env_name, robot, control_freq, num_steps, created_at

    Args:
        episode: The episode to write.
        path: Destination file path (.hdf5).

    Returns:
        The resolved Path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")

        # Images — gzip compressed
        obs.create_dataset("rgb_scene", data=episode.rgb_scenes, compression="gzip", compression_opts=4)
        obs.create_dataset("rgb_wrist", data=episode.rgb_wrists, compression="gzip", compression_opts=4)

        # State
        obs.create_dataset("qpos", data=episode.qpos_array)
        obs.create_dataset("qvel", data=episode.qvel_array)
        obs.create_dataset("gripper", data=episode.gripper_array)
        obs.create_dataset("ee_pose", data=episode.ee_poses)

        # Optional object pose
        obj_poses = episode.object_poses
        if obj_poses is not None:
            obs.create_dataset("object_pose", data=obj_poses)

        # Optional contacts (variable-length per step)
        contacts = episode.contacts_list
        has_contacts = any(c is not None for c in contacts)
        if has_contacts:
            cg = obs.create_group("contacts")
            for i, c in enumerate(contacts):
                if c is not None:
                    cg.create_dataset(f"step_{i:06d}", data=c)

        # Actions
        f.create_dataset("action", data=episode.actions)

        # Metadata as attrs
        meta = episode.metadata
        f.attrs["seed"] = meta.seed if meta.seed is not None else -1
        f.attrs["env_name"] = meta.env_name
        f.attrs["robot"] = meta.robot
        f.attrs["control_freq"] = meta.control_freq
        f.attrs["num_steps"] = len(episode)
        f.attrs["created_at"] = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        # Extra metadata
        for k, v in meta.extra.items():
            f.attrs[f"extra/{k}"] = v

    return path


def episode_path(output_dir: str | Path, episode_idx: int) -> Path:
    """Generate the canonical episode file path.

    Args:
        output_dir: Base directory for episodes.
        episode_idx: Zero-based episode index.

    Returns:
        Path like ``output_dir/ep_000042.hdf5``.
    """
    return Path(output_dir) / f"ep_{episode_idx:06d}.hdf5"
