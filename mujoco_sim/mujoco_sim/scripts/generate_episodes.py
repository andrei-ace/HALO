"""CLI entry point for teacher episode generation.

Usage::

    python -m mujoco_sim.scripts.generate_episodes --num-episodes 10 --output-dir episodes
"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PICK demo episodes with scripted teacher")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to generate")
    parser.add_argument("--output-dir", type=str, default="episodes", help="Output directory for HDF5 files")
    parser.add_argument("--seed-base", type=int, default=0, help="Base seed (episode i uses seed_base + i)")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--control-freq", type=int, default=20, help="Control frequency (Hz)")
    parser.add_argument("--stabilize", type=float, default=5.0, help="Seconds of zero-action settling before recording")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from mujoco_sim.config import EnvConfig
    from mujoco_sim.runner import run_teacher

    env_config = EnvConfig(control_freq=args.control_freq)
    paths = run_teacher(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed_base=args.seed_base,
        env_config=env_config,
        max_steps=args.max_steps,
        stabilize_seconds=args.stabilize,
    )

    print(f"Generated {len(paths)} episodes in {args.output_dir}/")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
