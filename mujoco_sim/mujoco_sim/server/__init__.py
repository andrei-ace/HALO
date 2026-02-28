"""MuJoCo sim server — standalone process owning SO101Env + PickTeacher.

Exposes 2 ZMQ channels:
    TelemetryStream (PUB): frames + state at render_fps
    CommandRPC (REP): step, reset, teacher_step, configure, set_hint, shutdown

Single-threaded main loop (required for macOS OpenGL rendering).
"""

from __future__ import annotations

import logging
import time

import msgpack
import numpy as np
import zmq

from mujoco_sim.server.config import SimServerConfig
from mujoco_sim.server.handlers import dispatch_command
from mujoco_sim.server.protocol import pack_telemetry

logger = logging.getLogger(__name__)


class SimServer:
    """MuJoCo sim server process.

    Owns SO101Env and PickTeacher. Runs a single-threaded polling loop
    that handles commands on CommandRPC and publishes telemetry on
    TelemetryStream at the
    configured render rate.
    """

    def __init__(self, config: SimServerConfig | None = None) -> None:
        self._config = config or SimServerConfig()
        self._running = False
        self._teacher_mode = False

        # ZMQ context + sockets (created in run())
        self._ctx: zmq.Context | None = None
        self._pub_telemetry: zmq.Socket | None = None
        self._rep_commands: zmq.Socket | None = None
        self._poller: zmq.Poller | None = None

        # Latest hint from HALO
        self._latest_hint: dict | None = None

        # Telemetry drop counter
        self._telemetry_drops: int = 0

        # Env + teacher (created in run())
        self._env = None
        self._teacher = None

    def run(self) -> None:
        """Run the server main loop. Blocks until shutdown command received."""
        from mujoco_sim.env import SO101Env
        from mujoco_sim.teacher.pick_teacher import PickTeacher

        cfg = self._config

        # Create env + teacher (must be on main thread for macOS OpenGL)
        logger.info("Creating SO101Env (scene=%s)", cfg.env_config.scene_xml)
        self._env = SO101Env(cfg.env_config)
        self._teacher = PickTeacher(cfg.teacher_config)

        # Create ZMQ sockets
        self._ctx = zmq.Context()

        # TelemetryStream: PUB telemetry
        self._pub_telemetry = self._ctx.socket(zmq.PUB)
        self._pub_telemetry.bind(cfg.telemetry_url)
        logger.info("TelemetryStream PUB bound to %s", cfg.telemetry_url)

        # CommandRPC: REP commands
        self._rep_commands = self._ctx.socket(zmq.REP)
        self._rep_commands.bind(cfg.command_url)
        logger.info("CommandRPC REP bound to %s", cfg.command_url)

        # Poller for non-blocking recv
        self._poller = zmq.Poller()
        self._poller.register(self._rep_commands, zmq.POLLIN)

        # Reset env
        self._env.reset(seed=0)

        render_interval = 1.0 / cfg.render_fps
        last_render = 0.0
        step_count = 0
        self._running = True

        logger.info("SimServer ready — entering main loop (render_fps=%d)", cfg.render_fps)

        try:
            while self._running:
                # 1. Poll CommandRPC — 10ms timeout
                socks = dict(self._poller.poll(timeout=10))

                if self._rep_commands in socks:
                    raw = self._rep_commands.recv()
                    msg = msgpack.unpackb(raw, raw=False)
                    # Handle configure specially to update teacher_mode
                    if msg.get("type") == "configure":
                        if "teacher_mode" in msg:
                            self._teacher_mode = msg["teacher_mode"]
                            logger.info("teacher_mode = %s", self._teacher_mode)
                    reply, shutdown = dispatch_command(msg, self._env, self._teacher, teacher_mode=self._teacher_mode)

                    # Store hint from handler response (single source of truth)
                    if "hint" in reply:
                        self._latest_hint = reply.pop("hint")
                    self._rep_commands.send(msgpack.packb(reply, use_bin_type=True))

                    if msg.get("type") == "step" and reply.get("type") == "step_ok":
                        step_count += 1
                    elif msg.get("type") == "teacher_step" and reply.get("type") == "teacher_step_ok":
                        step_count += 1

                    if shutdown:
                        break

                # 2. Publish telemetry at render_fps
                now = time.monotonic()
                if now - last_render >= render_interval:
                    last_render = now
                    self._publish_telemetry(step_count)

        finally:
            self._cleanup()

    def _publish_telemetry(self, step_count: int) -> None:
        """Render current state and publish on TelemetryStream."""
        obs = self._env._extract_obs()  # noqa: SLF001
        phase_id = self._teacher.phase if self._teacher_mode else 0
        done = self._teacher.done if self._teacher_mode else False

        # Get last action from env ctrl
        action = np.array(self._env.mujoco_data.ctrl[:6], copy=True)

        msg = pack_telemetry(
            ts_ms=int(time.monotonic() * 1000),
            step_count=step_count,
            phase_id=phase_id,
            done=done,
            qpos=obs["qpos"],
            qvel=obs["qvel"],
            ee_pose=obs["ee_pose"],
            object_pose=obs["object_pose"],
            joint_pos=obs["joint_pos"],
            gripper=float(obs["gripper"]),
            action=action,
            rgb_scene=obs["rgb_scene"],
            rgb_wrist=obs["rgb_wrist"],
            jpeg_quality=self._config.jpeg_quality,
        )
        packed = msgpack.packb(msg, use_bin_type=True)
        try:
            self._pub_telemetry.send(packed, zmq.NOBLOCK)
        except zmq.ZMQError as exc:
            self._telemetry_drops += 1
            if self._telemetry_drops == 1:
                logger.warning("TelemetryStream send skipped (first drop): %s", exc)
            else:
                logger.debug("TelemetryStream send skipped (drop #%d): %s", self._telemetry_drops, exc)

    def _cleanup(self) -> None:
        """Close sockets and ZMQ context."""
        self._running = False
        for sock in (self._pub_telemetry, self._rep_commands):
            if sock is not None:
                sock.close(linger=100)
        if self._ctx is not None:
            self._ctx.term()
        if self._env is not None:
            self._env.close()
        logger.info("SimServer shut down")
