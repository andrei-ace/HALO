"""HALO-side ZMQ client for communicating with the MuJoCo sim server.

Connects to 2 channels:
    TelemetryStream (SUB): frames + state receiver (background thread)
    CommandRPC (REQ): step, reset, teacher_step, configure, set_hint

Usage::

    from halo.bridge.config import SimBridgeConfig
    from halo.bridge.sim_client import SimClient

    client = SimClient(SimBridgeConfig())
    client.start()
    client.reset(seed=42)
    for _ in range(100):
        client.step(home_action)
    resp = client.teacher_step()
    client.shutdown()
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
import warnings
from typing import Any

import msgpack
import numpy as np
import zmq

from halo.bridge import BridgeTransportError
from halo.bridge.config import SimBridgeConfig

logger = logging.getLogger(__name__)


class SimClient:
    """HALO-side ZMQ client for the MuJoCo sim server.

    Manages connections to telemetry + command channels, with a background thread
    for receiving telemetry.

    Supports two start modes:
        - Standalone: server already running, just connect.
        - Managed: spawn the server subprocess, wait for readiness, then connect.
    """

    def __init__(self, config: SimBridgeConfig | None = None) -> None:
        self._config = config or SimBridgeConfig()
        self._started = False

        # ZMQ
        self._ctx: zmq.Context | None = None
        self._sub_telemetry: zmq.Socket | None = None
        self._req_commands: zmq.Socket | None = None

        # Telemetry receiver thread
        self._recv_thread: threading.Thread | None = None
        self._recv_stop = threading.Event()
        self._latest_telemetry: dict | None = None
        self._telemetry_lock = threading.Lock()
        self._telemetry_ready = threading.Event()

        # Managed subprocess
        self._server_proc: subprocess.Popen | None = None

        # Command lock (REQ/REP is strictly sequential)
        self._cmd_lock = threading.Lock()
        self._retryable_commands = {"reset", "get_state", "set_state", "configure", "set_hint"}

    @property
    def started(self) -> bool:
        return self._started

    @property
    def latest_telemetry(self) -> dict | None:
        """Latest decoded telemetry message (thread-safe read)."""
        with self._telemetry_lock:
            return self._latest_telemetry

    def start(self, timeout: float | None = None) -> None:
        """Connect to the sim server.

        Args:
            timeout: Seconds to wait for first telemetry frame.
                Default: config.server_startup_timeout_s for managed, 10s for standalone.

        Raises:
            TimeoutError: If first telemetry not received within timeout.
            BridgeTransportError: If managed server fails to start.
        """
        if self._started:
            return

        cfg = self._config

        if cfg.managed:
            self._start_managed()

        if timeout is None:
            timeout = cfg.server_startup_timeout_s if cfg.managed else 10.0

        # Create ZMQ context + sockets
        self._ctx = zmq.Context()

        # TelemetryStream: SUB telemetry (connect to server PUB)
        self._sub_telemetry = self._ctx.socket(zmq.SUB)
        self._sub_telemetry.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub_telemetry.setsockopt(zmq.RCVTIMEO, cfg.recv_timeout_ms)
        self._sub_telemetry.connect(cfg.telemetry_url)
        logger.info("TelemetryStream SUB connected to %s", cfg.telemetry_url)

        # CommandRPC: REQ commands (connect to server REP)
        self._connect_command_socket()

        # Start telemetry receiver thread
        self._recv_stop.clear()
        self._telemetry_ready.clear()
        self._recv_thread = threading.Thread(target=self._telemetry_loop, daemon=True, name="sim-telemetry")
        self._recv_thread.start()

        # Wait for first telemetry
        if not self._telemetry_ready.wait(timeout=timeout):
            self.stop()
            raise TimeoutError(f"No telemetry received within {timeout}s from {cfg.telemetry_url}")

        self._started = True
        logger.info("SimClient started (telemetry streaming)")

    def stop(self) -> None:
        """Disconnect and clean up."""
        self._recv_stop.set()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None

        for sock in (self._sub_telemetry, self._req_commands):
            if sock is not None:
                sock.close(linger=100)
        self._sub_telemetry = None
        self._req_commands = None

        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

        if self._server_proc is not None:
            self._stop_managed()

        self._started = False
        logger.info("SimClient stopped")

    # ------------------------------------------------------------------
    # CommandRPC methods
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> dict:
        """Step the environment with a 6D action.

        Returns:
            Dict with keys: type, reward, done.
        """
        from mujoco_sim.server.protocol import CMD_STEP, ndarray_to_bytes

        return self._send_command(
            {
                "type": CMD_STEP,
                "action": ndarray_to_bytes(np.asarray(action, dtype=np.float64)),
            }
        )

    def reset(self, seed: int | None = None) -> dict:
        """Reset environment and teacher.

        Returns:
            Dict with keys: type, seed.
        """
        from mujoco_sim.server.protocol import CMD_RESET

        return self._send_command({"type": CMD_RESET, "seed": seed})

    def get_state(self) -> dict:
        """Get full MuJoCo state (qpos, qvel)."""
        from mujoco_sim.server.protocol import CMD_GET_STATE

        return self._send_command({"type": CMD_GET_STATE})

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> dict:
        """Inject MuJoCo state."""
        from mujoco_sim.server.protocol import CMD_SET_STATE, ndarray_to_bytes

        return self._send_command(
            {
                "type": CMD_SET_STATE,
                "qpos": ndarray_to_bytes(qpos),
                "qvel": ndarray_to_bytes(qvel),
            }
        )

    def teacher_step(self) -> dict:
        """Run one teacher step (server-side teacher policy).

        Returns:
            Dict with keys: type, action (bytes), phase_id, done,
            qpos (bytes), qvel (bytes), ee_pose (bytes), object_pose (bytes),
            joint_pos (bytes), gripper, rgb_scene (bytes), rgb_wrist (bytes).
        """
        from mujoco_sim.server.protocol import CMD_TEACHER_STEP

        return self._send_command({"type": CMD_TEACHER_STEP})

    def configure(self, **kwargs: Any) -> dict:
        """Send runtime configuration changes.

        Args:
            teacher_mode: Enable/disable teacher stepping.
            telemetry_profile: Optional telemetry profile selector.
        """
        from mujoco_sim.server.protocol import CMD_CONFIGURE

        return self._send_command({"type": CMD_CONFIGURE, **kwargs})

    def shutdown(self) -> dict:
        """Send shutdown command to server."""
        from mujoco_sim.server.protocol import CMD_SHUTDOWN

        resp = self._send_command({"type": CMD_SHUTDOWN})
        self.stop()
        return resp

    # ------------------------------------------------------------------
    # Hint updates (sent as CommandRPC command in protocol v2)
    # ------------------------------------------------------------------

    def publish_hint(
        self,
        *,
        target_handle: str | None = None,
        bbox_xywh: tuple[int, int, int, int] | None = None,
        confidence: float = 0.0,
        tracker_ok: bool = False,
    ) -> None:
        """Deprecated alias for set_hint()."""
        warnings.warn(
            "publish_hint() is deprecated in protocol v2; use set_hint()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_hint(
            target_handle=target_handle,
            bbox_xywh=bbox_xywh,
            confidence=confidence,
            tracker_ok=tracker_ok,
        )

    def set_hint(
        self,
        *,
        target_handle: str | None = None,
        bbox_xywh: tuple[int, int, int, int] | None = None,
        confidence: float = 0.0,
        tracker_ok: bool = False,
    ) -> dict:
        """Send a tracking hint update to the sim over CommandRPC."""
        from mujoco_sim.server.protocol import CMD_SET_HINT

        return self._send_command(
            {
                "type": CMD_SET_HINT,
                "ts_ms": int(time.monotonic() * 1000),
                "target_handle": target_handle,
                "bbox_xywh": list(bbox_xywh) if bbox_xywh is not None else None,
                "confidence": confidence,
                "tracker_ok": tracker_ok,
            }
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send_command(self, msg: dict) -> dict:
        """Send a command on CommandRPC and wait for response (thread-safe)."""
        if self._req_commands is None:
            raise BridgeTransportError("SimClient not started")
        with self._cmd_lock:
            try:
                return self._send_command_once(msg)
            except zmq.ZMQError as exc:
                cmd_type = str(msg.get("type", "unknown"))
                self._reset_command_socket()
                if cmd_type not in self._retryable_commands:
                    raise BridgeTransportError(f"CommandRPC failed ({cmd_type}): {exc}") from exc

                try:
                    return self._send_command_once(msg)
                except zmq.ZMQError as retry_exc:
                    self._reset_command_socket()
                    raise BridgeTransportError(f"CommandRPC retry failed ({cmd_type}): {retry_exc}") from retry_exc

    def _send_command_once(self, msg: dict) -> dict:
        if self._req_commands is None:
            raise BridgeTransportError("SimClient command socket unavailable")

        self._req_commands.send(msgpack.packb(msg, use_bin_type=True))
        raw = self._req_commands.recv()
        resp = msgpack.unpackb(raw, raw=False)
        if resp.get("type") == "error":
            raise BridgeTransportError(f"Server error: {resp.get('message', 'unknown')}")
        return resp

    def _connect_command_socket(self) -> None:
        if self._ctx is None:
            raise BridgeTransportError("SimClient context not initialized")

        cfg = self._config
        req = self._ctx.socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, cfg.command_timeout_ms)
        req.setsockopt(zmq.SNDTIMEO, cfg.command_timeout_ms)
        req.connect(cfg.command_url)
        self._req_commands = req
        logger.info("CommandRPC REQ connected to %s", cfg.command_url)

    def _reset_command_socket(self) -> None:
        old_sock = self._req_commands
        self._req_commands = None
        if old_sock is not None:
            old_sock.close(linger=0)
        self._connect_command_socket()

    def _telemetry_loop(self) -> None:
        """Background thread: receive telemetry on TelemetryStream."""
        from mujoco_sim.server.protocol import unpack_telemetry

        while not self._recv_stop.is_set():
            try:
                raw = self._sub_telemetry.recv()
                msg = msgpack.unpackb(raw, raw=False)
                decoded = unpack_telemetry(msg)
                with self._telemetry_lock:
                    self._latest_telemetry = decoded
                if not self._telemetry_ready.is_set():
                    self._telemetry_ready.set()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._recv_stop.is_set():
                    break
                continue

    def _start_managed(self) -> None:
        """Spawn the sim server as a subprocess."""
        cmd = [sys.executable, "-m", "mujoco_sim.server"]
        cfg = self._config
        # Parse ports from URLs
        cmd.extend(["--telemetry-port", str(self._parse_port(cfg.telemetry_url))])
        cmd.extend(["--command-port", str(self._parse_port(cfg.command_url))])

        logger.info("Starting managed sim server: %s", " ".join(cmd))
        self._server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Give the server a moment to bind sockets
        time.sleep(1.0)

        if self._server_proc.poll() is not None:
            raise BridgeTransportError("Managed server exited immediately")

    def _stop_managed(self) -> None:
        """Stop the managed server subprocess."""
        if self._server_proc is None:
            return
        self._server_proc.terminate()
        try:
            self._server_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._server_proc.kill()
            self._server_proc.wait()
        self._server_proc = None

    @staticmethod
    def _parse_port(url: str) -> int:
        """Extract port from a tcp://host:port URL."""
        return int(url.rsplit(":", 1)[-1])
