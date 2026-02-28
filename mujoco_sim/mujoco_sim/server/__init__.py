"""MuJoCo sim server — standalone process owning SO101Env + PickTeacher.

Exposes 4 ZMQ channels:
    Ch1 (PUB): telemetry — frames + state at render_fps
    Ch2 (SUB): tracking hints from HALO
    Ch3 (REP): commands — step, reset, teacher_step, etc.
    Ch4 (REQ): queries — VLM detect, tracker init/update (to HALO)

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
    that handles commands on Ch3, receives hints on Ch2, and publishes
    telemetry on Ch1 at the configured render rate.
    """

    def __init__(self, config: SimServerConfig | None = None) -> None:
        self._config = config or SimServerConfig()
        self._running = False
        self._teacher_mode = False

        # ZMQ context + sockets (created in run())
        self._ctx: zmq.Context | None = None
        self._pub_telemetry: zmq.Socket | None = None
        self._sub_hints: zmq.Socket | None = None
        self._rep_commands: zmq.Socket | None = None
        self._req_queries: zmq.Socket | None = None
        self._poller: zmq.Poller | None = None

        # Latest hint from HALO
        self._latest_hint: dict | None = None

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

        # Ch1: PUB telemetry
        self._pub_telemetry = self._ctx.socket(zmq.PUB)
        self._pub_telemetry.bind(cfg.telemetry_url)
        logger.info("Ch1 PUB (telemetry) bound to %s", cfg.telemetry_url)

        # Ch2: SUB hints
        self._sub_hints = self._ctx.socket(zmq.SUB)
        self._sub_hints.bind(cfg.hints_url)
        self._sub_hints.setsockopt(zmq.SUBSCRIBE, b"")
        logger.info("Ch2 SUB (hints) bound to %s", cfg.hints_url)

        # Ch3: REP commands
        self._rep_commands = self._ctx.socket(zmq.REP)
        self._rep_commands.bind(cfg.command_url)
        logger.info("Ch3 REP (commands) bound to %s", cfg.command_url)

        # Ch4: REQ queries (connect — HALO binds the REP side)
        self._req_queries = self._ctx.socket(zmq.REQ)
        self._req_queries.setsockopt(zmq.RCVTIMEO, cfg.query_timeout_ms)
        self._req_queries.setsockopt(zmq.SNDTIMEO, cfg.query_timeout_ms)
        # Don't connect yet — only connect when HALO's query service is available
        self._query_connected = False

        # Poller for non-blocking recv
        self._poller = zmq.Poller()
        self._poller.register(self._rep_commands, zmq.POLLIN)
        self._poller.register(self._sub_hints, zmq.POLLIN)

        # Reset env
        self._env.reset(seed=0)

        render_interval = 1.0 / cfg.render_fps
        last_render = 0.0
        step_count = 0
        self._running = True

        logger.info("SimServer ready — entering main loop (render_fps=%d)", cfg.render_fps)

        try:
            while self._running:
                # 1. Poll Ch3 (commands) + Ch2 (hints) — 10ms timeout
                socks = dict(self._poller.poll(timeout=10))

                if self._rep_commands in socks:
                    raw = self._rep_commands.recv()
                    msg = msgpack.unpackb(raw, raw=False)
                    # Handle configure specially to update teacher_mode
                    if msg.get("type") == "configure":
                        if "teacher_mode" in msg:
                            self._teacher_mode = msg["teacher_mode"]
                            logger.info("teacher_mode = %s", self._teacher_mode)
                        if "query_url" in msg:
                            # Connect Ch4 to HALO's query service
                            if not self._query_connected:
                                query_url = msg["query_url"]
                                self._req_queries.connect(query_url)
                                self._query_connected = True
                                logger.info("Ch4 REQ (queries) connected to %s", query_url)

                    reply, shutdown = dispatch_command(msg, self._env, self._teacher, teacher_mode=self._teacher_mode)
                    self._rep_commands.send(msgpack.packb(reply, use_bin_type=True))

                    if msg.get("type") == "step" and reply.get("type") == "step_ok":
                        step_count += 1
                    elif msg.get("type") == "teacher_step" and reply.get("type") == "teacher_step_ok":
                        step_count += 1

                    if shutdown:
                        break

                if self._sub_hints in socks:
                    raw = self._sub_hints.recv(zmq.NOBLOCK)
                    self._latest_hint = msgpack.unpackb(raw, raw=False)

                # 2. Publish telemetry at render_fps
                now = time.monotonic()
                if now - last_render >= render_interval:
                    last_render = now
                    self._publish_telemetry(step_count)

        finally:
            self._cleanup()

    def _publish_telemetry(self, step_count: int) -> None:
        """Render current state and publish on Ch1."""
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
        self._pub_telemetry.send(packed, zmq.NOBLOCK)

    def query_halo(self, request: dict) -> dict | None:
        """Send a query on Ch4 (REQ) and wait for response.

        Returns None if not connected or on timeout.
        """
        if not self._query_connected or self._req_queries is None:
            return None
        try:
            self._req_queries.send(msgpack.packb(request, use_bin_type=True))
            raw = self._req_queries.recv()
            return msgpack.unpackb(raw, raw=False)
        except zmq.ZMQError as exc:
            logger.warning("Ch4 query failed: %s", exc)
            return None

    def _cleanup(self) -> None:
        """Close sockets and ZMQ context."""
        self._running = False
        for sock in (self._pub_telemetry, self._sub_hints, self._rep_commands, self._req_queries):
            if sock is not None:
                sock.close(linger=100)
        if self._ctx is not None:
            self._ctx.term()
        if self._env is not None:
            self._env.close()
        logger.info("SimServer shut down")
