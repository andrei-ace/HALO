"""CLI entry point: ``python -m mujoco_sim.server``."""

from __future__ import annotations

import argparse
import logging

from mujoco_sim.config import EnvConfig
from mujoco_sim.server import SimServer
from mujoco_sim.server.config import SimServerConfig
from mujoco_sim.teacher.pick_teacher import TeacherConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo sim server for HALO")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--telemetry-port", type=int, default=5560, help="Ch1 PUB port (default: 5560)")
    parser.add_argument("--hints-port", type=int, default=5561, help="Ch2 SUB port (default: 5561)")
    parser.add_argument("--command-port", type=int, default=5562, help="Ch3 REP port (default: 5562)")
    parser.add_argument("--query-port", type=int, default=5563, help="Ch4 REQ port (default: 5563)")
    parser.add_argument("--render-fps", type=int, default=10, help="Telemetry publish rate (default: 10)")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality 0-100 (default: 85)")
    parser.add_argument("--control-freq", type=int, default=20, help="Env control frequency (default: 20)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    env_config = EnvConfig(control_freq=args.control_freq)
    teacher_config = TeacherConfig()

    config = SimServerConfig(
        host=args.host,
        telemetry_port=args.telemetry_port,
        hints_port=args.hints_port,
        command_port=args.command_port,
        query_port=args.query_port,
        render_fps=args.render_fps,
        jpeg_quality=args.jpeg_quality,
        env_config=env_config,
        teacher_config=teacher_config,
    )

    server = SimServer(config)
    server.run()


if __name__ == "__main__":
    main()
