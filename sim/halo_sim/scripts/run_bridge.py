"""Launch sim server for HALO connection via ZeroMQ.

Usage:
    python -m halo_sim.scripts.run_bridge --action-port 5555 --obs-port 5556

Note: Requires Isaac Lab runtime.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def run_bridge(action_port: int = 5555, obs_port: int = 5556) -> None:
    """Start the ZeroMQ bridge server.

    **STUB** — requires Isaac Lab runtime.

    When implemented, this function will:
      1. Create Isaac Lab PickEnv (single env mode)
      2. Create SimServer with the env
      3. Run the server (blocks until Ctrl+C)

    Raises ``NotImplementedError`` until wired to a live Isaac Lab session.
    """
    raise NotImplementedError(
        f"run_bridge is a stub — Isaac Lab runtime required. Requested action_port={action_port}, obs_port={obs_port}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch HALO-Isaac Lab ZeroMQ bridge")
    parser.add_argument("--action-port", type=int, default=5555)
    parser.add_argument("--obs-port", type=int, default=5556)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_bridge(action_port=args.action_port, obs_port=args.obs_port)


if __name__ == "__main__":
    main()
