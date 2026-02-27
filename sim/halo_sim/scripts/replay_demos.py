"""Replay verification — sample episodes, replay actions, measure EE deviation.

Usage:
    python -m halo_sim.scripts.replay_demos --dataset datasets/demos_v0_manifest.json --samples 100
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def replay_demos(
    manifest_path: str,
    num_samples: int = 100,
) -> None:
    """Sample episodes from shards, replay actions with same seed, measure max EE deviation.

    Pass criterion: <5mm max EE deviation.

    **STUB** — requires Isaac Lab runtime to re-execute actions in the
    simulator and compare EE trajectories.

    When implemented, this function will:
      1. Load manifest and sample episodes from shards
      2. Re-create each env with the recorded seed
      3. Replay recorded actions and compare EE positions
      4. Report per-episode pass/fail (threshold: 5 mm max deviation)

    Raises ``NotImplementedError`` until wired to a live Isaac Lab session.
    """
    raise NotImplementedError(
        f"replay_demos is a stub — Isaac Lab runtime required. Manifest: {manifest_path!r}, samples: {num_samples}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay demonstration episodes for verification")
    parser.add_argument("--dataset", type=str, required=True, help="Path to manifest JSON")
    parser.add_argument("--samples", type=int, default=100, help="Number of episodes to sample")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    replay_demos(manifest_path=args.dataset, num_samples=args.samples)


if __name__ == "__main__":
    main()
