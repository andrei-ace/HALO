"""Episode generation loop: reset env → stabilize → run teacher → write HDF5."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mujoco_sim.config import EnvConfig
from mujoco_sim.dataset import EpisodeMetadata, RawEpisode, Timestep, episode_path, write_episode
from mujoco_sim.teacher.pick_teacher import PickTeacher, TeacherConfig

logger = logging.getLogger(__name__)

# Default stabilization: 5 seconds of zero-action steps to let physics settle
DEFAULT_STABILIZE_SECONDS = 5.0


def run_teacher(
    num_episodes: int,
    output_dir: str | Path,
    *,
    seed_base: int = 0,
    env_config: EnvConfig | None = None,
    teacher_config: TeacherConfig | None = None,
    max_steps: int = 500,
    stabilize_seconds: float = DEFAULT_STABILIZE_SECONDS,
) -> list[Path]:
    """Generate episodes by running the scripted teacher in the environment.

    Each episode begins with a stabilization phase (zero-action steps for
    ``stabilize_seconds``) to let physics settle before recording starts.

    Args:
        num_episodes: Number of episodes to generate.
        output_dir: Directory to write HDF5 files.
        seed_base: Base seed; episode *i* uses ``seed_base + i``.
        env_config: Environment configuration (default: ``EnvConfig()``).
        teacher_config: Teacher gains/thresholds (default: ``TeacherConfig()``).
        max_steps: Maximum steps per episode (safety limit).
        stabilize_seconds: Seconds of zero-action settling before recording.

    Returns:
        List of paths to written HDF5 files.
    """
    # Lazy import to avoid requiring robosuite at module level
    from mujoco_sim.env import RobosuiteEnv

    env_config = env_config or EnvConfig()
    teacher_config = teacher_config or TeacherConfig()
    output_dir = Path(output_dir)

    env = RobosuiteEnv(env_config)
    teacher = PickTeacher(teacher_config)
    written: list[Path] = []

    try:
        for ep_idx in range(num_episodes):
            seed = seed_base + ep_idx
            path = _run_single_episode(
                env=env,
                teacher=teacher,
                ep_idx=ep_idx,
                seed=seed,
                output_dir=output_dir,
                env_config=env_config,
                max_steps=max_steps,
                stabilize_steps=int(stabilize_seconds * env_config.control_freq),
            )
            written.append(path)
            logger.info("Episode %d/%d → %s", ep_idx + 1, num_episodes, path)
    finally:
        env.close()

    return written


def _run_single_episode(
    *,
    env,
    teacher: PickTeacher,
    ep_idx: int,
    seed: int,
    output_dir: Path,
    env_config: EnvConfig,
    max_steps: int,
    stabilize_steps: int,
) -> Path:
    """Run one episode and write to HDF5."""
    obs = env.reset(seed=seed)

    # --- Stabilization phase: step with zero actions to let physics settle ---
    zero_action = np.zeros(7)
    for _ in range(stabilize_steps):
        obs, _, _, _ = env.step(zero_action)
    logger.debug("Stabilized for %d steps (seed=%d)", stabilize_steps, seed)

    # --- Recording phase ---
    teacher.reset()

    meta = EpisodeMetadata(
        seed=seed,
        env_name=env_config.env_name,
        robot=env_config.robot,
        control_freq=env_config.control_freq,
        extra={"teacher": "pick", "stabilize_steps": str(stabilize_steps)},
    )
    episode = RawEpisode(metadata=meta)

    for step_idx in range(max_steps):
        action, phase_id, done = teacher.step(obs)

        # Record BEFORE stepping (obs is current, action is what we're about to apply)
        ts = Timestep(
            rgb_scene=obs["rgb_scene"],
            rgb_wrist=obs["rgb_wrist"],
            qpos=obs["qpos"],
            qvel=obs["qvel"],
            gripper=float(obs["gripper"]),
            ee_pose=obs["ee_pose"],
            action=np.array(action, copy=True),
            phase_id=phase_id,
            object_pose=obs.get("object_pose"),
        )
        episode.append(ts)

        if done:
            break

        obs, _reward, env_done, _info = env.step(action)
        if env_done:
            break

    path = episode_path(output_dir, ep_idx)
    write_episode(episode, path)
    return path
