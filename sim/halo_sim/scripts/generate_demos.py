"""Generate teacher demonstrations -> sharded HDF5.

Usage:
    python -m halo_sim.scripts.generate_demos --num-episodes 10000 --num-envs 64
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def generate_demos(
    num_episodes: int = 10000,
    num_envs: int = 64,
    output_dir: str = "datasets",
    max_steps: int = 300,
) -> None:
    """Generate demonstration episodes using the teacher pipeline.

    **STUB** — requires Isaac Lab runtime for physics.

    When implemented, this function will:
      1. Create Isaac Lab PickEnv with ``PickEnvCfg(num_envs=num_envs)``
      2. Run TeacherFSM + IKTeacher to generate privileged-state actions
      3. Record episodes to sharded HDF5 via ``ShardedRecorder``

    Raises ``NotImplementedError`` until wired to a live Isaac Lab session.
    """
    raise NotImplementedError(
        "generate_demos is a stub — Isaac Lab runtime required. "
        f"Requested {num_episodes} episodes across {num_envs} envs to {output_dir!r}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher demonstrations")
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_demos(
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
