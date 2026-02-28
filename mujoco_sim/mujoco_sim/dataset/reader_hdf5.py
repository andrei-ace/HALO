"""HDF5 episode reader — inverse of writer_hdf5."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mujoco_sim.dataset.raw_episode import EpisodeMetadata, RawEpisode, Timestep


def read_episode(path: str | Path) -> RawEpisode:
    """Load an HDF5 episode file into a RawEpisode.

    Reads the layout written by :func:`writer_hdf5.write_episode`.

    Args:
        path: Path to the ``.hdf5`` file.

    Returns:
        A fully-populated RawEpisode with metadata.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        # Metadata
        seed_val = int(f.attrs["seed"])
        meta = EpisodeMetadata(
            seed=seed_val if seed_val >= 0 else None,
            env_name=str(f.attrs["env_name"]),
            robot=str(f.attrs["robot"]),
            control_freq=int(f.attrs["control_freq"]),
        )

        # Extra metadata
        for key in f.attrs:
            if key.startswith("extra/"):
                meta.extra[key.removeprefix("extra/")] = f.attrs[key]

        episode = RawEpisode(metadata=meta)

        # Load arrays
        obs = f["obs"]
        rgb_scenes = obs["rgb_scene"][:]  # (T, H, W, 3)
        rgb_wrists = obs["rgb_wrist"][:]
        qpos_arr = obs["qpos"][:]
        qvel_arr = obs["qvel"][:]
        gripper_arr = obs["gripper"][:]
        ee_poses = obs["ee_pose"][:]
        actions = f["action"][:]

        has_object = "object_pose" in obs
        object_poses = obs["object_pose"][:] if has_object else None

        has_contacts = "contacts" in obs
        contacts_group = obs["contacts"] if has_contacts else None

        num_steps = rgb_scenes.shape[0]
        for i in range(num_steps):
            obj_pose = object_poses[i] if object_poses is not None else None

            contacts = None
            if contacts_group is not None:
                key = f"step_{i:06d}"
                if key in contacts_group:
                    contacts = np.array(contacts_group[key])

            ts = Timestep(
                rgb_scene=rgb_scenes[i],
                rgb_wrist=rgb_wrists[i],
                qpos=qpos_arr[i],
                qvel=qvel_arr[i],
                gripper=float(gripper_arr[i]),
                ee_pose=ee_poses[i],
                action=actions[i],
                object_pose=obj_pose,
                contacts=contacts,
            )
            episode.append(ts)

    return episode
