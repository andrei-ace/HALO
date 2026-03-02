"""MuJoCo sim server — standalone process owning SO101Env.

Exposes 2 ZMQ channels:
    TelemetryStream (PUB): frames + state at render_fps
    CommandRPC (REP): step, reset, start_pick, configure, set_hint, shutdown

Physics runs autonomously at physics_hz (default 20Hz). When a trajectory
is active (from start_pick), the loop samples it and steps the env.
Otherwise the arm holds its current position.

Single-threaded main loop (required for macOS OpenGL rendering).
"""

from __future__ import annotations

import logging
import time

import msgpack
import numpy as np
import zmq

from mujoco_sim.server.config import SimServerConfig
from mujoco_sim.server.handlers import ServerState, dispatch_command
from mujoco_sim.server.protocol import pack_telemetry

logger = logging.getLogger(__name__)


class SimServer:
    """MuJoCo sim server process.

    Owns SO101Env. Runs a single-threaded polling loop that:
    1. Polls CommandRPC (non-blocking)
    2. Steps physics at physics_hz (trajectory sampling or hold)
    3. Publishes telemetry at render_fps
    """

    def __init__(self, config: SimServerConfig | None = None) -> None:
        self._config = config or SimServerConfig()
        self._running = False

        # ZMQ context + sockets (created in run())
        self._ctx: zmq.Context | None = None
        self._pub_telemetry: zmq.Socket | None = None
        self._rep_commands: zmq.Socket | None = None
        self._poller: zmq.Poller | None = None

        # Latest hint from HALO
        self._latest_hint: dict | None = None

        # Telemetry drop counter
        self._telemetry_drops: int = 0

        # Env (created in run())
        self._env = None

        # Mutable server state (trajectory, phase tracking)
        self._state = ServerState()

    def run(self) -> None:
        """Run the server main loop. Blocks until shutdown command received."""
        from mujoco_sim.env import SO101Env

        cfg = self._config

        # Create env (must be on main thread for macOS OpenGL)
        logger.info("Creating SO101Env (scene=%s)", cfg.env_config.scene_xml)
        self._env = SO101Env(cfg.env_config)

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
        # Initialise hold target to home pose so IDLE holds position against gravity
        self._state.hold_target = self._env.home_qpos.copy()

        physics_interval = 1.0 / cfg.physics_hz
        render_interval = 1.0 / cfg.render_fps
        last_physics = 0.0
        last_render = 0.0
        step_count = 0
        self._running = True

        logger.info(
            "SimServer ready — entering main loop (physics_hz=%d, render_fps=%d)",
            cfg.physics_hz,
            cfg.render_fps,
        )

        try:
            while self._running:
                # 1. Poll CommandRPC — 1ms timeout (fast non-blocking)
                socks = dict(self._poller.poll(timeout=1))

                if self._rep_commands in socks:
                    raw = self._rep_commands.recv()
                    msg = msgpack.unpackb(raw, raw=False)
                    reply, shutdown = dispatch_command(msg, self._env, self._state)

                    # Store hint from handler response (single source of truth)
                    if "hint" in reply:
                        self._latest_hint = reply.pop("hint")
                    self._rep_commands.send(msgpack.packb(reply, use_bin_type=True))

                    if msg.get("type") == "step" and reply.get("type") == "step_ok":
                        step_count += 1

                    if shutdown:
                        break

                # 2. Autonomous physics at physics_hz
                now = time.monotonic()
                if now - last_physics >= physics_interval:
                    last_physics = now
                    self._physics_tick(now)
                    step_count += 1

                # 3. Publish telemetry at render_fps
                if now - last_render >= render_interval:
                    last_render = now
                    self._publish_telemetry(step_count)

        finally:
            self._cleanup()

    def _physics_tick(self, now: float) -> None:
        """One autonomous physics step: sample trajectory or hold position."""
        state = self._state
        env = self._env

        if state.trajectory is not None:
            t = now - state.traj_start_time
            if t >= state.trajectory.total_duration:
                # Trajectory complete — sample final point, latch as hold target
                arm_joints, gripper, phase_id = state.trajectory.sample(state.trajectory.total_duration)
                action = np.concatenate([arm_joints, [gripper]])
                env.step(action)
                state.hold_target = action.copy()
                state.phase_id = phase_id
                state.done = True
                state.trajectory = None
                logger.info("Trajectory complete (phase_id=%d)", phase_id)
            else:
                arm_joints, gripper, phase_id = state.trajectory.sample(t)
                action = np.concatenate([arm_joints, [gripper]])
                env.step(action)
                state.phase_id = phase_id
        else:
            # Hold fixed target position (prevents gravity drift)
            if state.hold_target is not None:
                env.step(state.hold_target)
            else:
                env.step(np.array(env.mujoco_data.ctrl[:6], copy=True))

    def _publish_telemetry(self, step_count: int) -> None:
        """Render current state and publish on TelemetryStream."""
        obs = self._env._extract_obs()  # noqa: SLF001

        # Get last action from env ctrl
        action = np.array(self._env.mujoco_data.ctrl[:6], copy=True)

        msg = pack_telemetry(
            ts_ms=int(time.monotonic() * 1000),
            step_count=step_count,
            phase_id=self._state.phase_id,
            done=self._state.done,
            qpos=obs["qpos"],
            qvel=obs["qvel"],
            ee_pose=obs["ee_pose"],
            object_pose=obs["object_pose"],
            red_object_pose=obs["red_object_pose"],
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
