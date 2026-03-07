"""CLI entry point: ``python -m mujoco_sim.server``."""

from __future__ import annotations

import argparse
import logging

from mujoco_sim.config import EnvConfig
from mujoco_sim.server import SimServer
from mujoco_sim.server.config import SimServerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo sim server for HALO")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--telemetry-port", type=int, default=5560, help="TelemetryStream PUB port (default: 5560)")
    parser.add_argument("--command-port", type=int, default=5561, help="CommandRPC REP port (default: 5561)")
    parser.add_argument("--render-fps", type=int, default=10, help="Telemetry publish rate (default: 10)")
    parser.add_argument("--physics-hz", type=int, default=20, help="Physics loop rate (default: 20)")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality 0-100 (default: 85)")
    parser.add_argument("--control-freq", type=int, default=20, help="Env control frequency (default: 20)")
    parser.add_argument("--log-file", default=None, help="Write logs to file instead of stderr")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    if args.log_file:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(args.log_file, mode="w")],
        )
    else:
        logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    env_config = EnvConfig(control_freq=args.control_freq)

    config = SimServerConfig(
        host=args.host,
        telemetry_port=args.telemetry_port,
        command_port=args.command_port,
        render_fps=args.render_fps,
        physics_hz=args.physics_hz,
        jpeg_quality=args.jpeg_quality,
        env_config=env_config,
    )

    server = SimServer(config)
    server.run()


if __name__ == "__main__":
    main()
