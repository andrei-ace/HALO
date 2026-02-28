"""Tests for SimClient and SimBridgeConfig.

Tests the client-side bridge code in isolation — no running server required
for config tests. Server integration tests use a real ZMQ server in-process.
"""

from __future__ import annotations

import threading
import time

import msgpack
import numpy as np
import pytest
import zmq
from mujoco_sim.server.protocol import (
    CMD_RESET,
    CMD_STEP,
    RESP_OK,
    RESP_RESET_OK,
    RESP_STEP_OK,
    pack_telemetry,
)

from halo.bridge import BridgeTransportError
from halo.bridge.config import SimBridgeConfig
from halo.bridge.sim_client import SimClient

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestSimBridgeConfig:
    def test_defaults(self):
        cfg = SimBridgeConfig()
        assert cfg.telemetry_url == "tcp://127.0.0.1:5560"
        assert cfg.hints_url == "tcp://127.0.0.1:5561"
        assert cfg.command_url == "tcp://127.0.0.1:5562"
        assert cfg.query_url == "tcp://127.0.0.1:5563"
        assert cfg.managed is False
        assert cfg.recv_timeout_ms == 5000
        assert cfg.command_timeout_ms == 10_000

    def test_custom_ports(self):
        cfg = SimBridgeConfig(
            telemetry_url="tcp://10.0.0.1:7000",
            command_url="tcp://10.0.0.1:7002",
        )
        assert cfg.telemetry_url == "tcp://10.0.0.1:7000"
        assert cfg.command_url == "tcp://10.0.0.1:7002"


# ---------------------------------------------------------------------------
# SimClient tests with a mock ZMQ server
# ---------------------------------------------------------------------------


def _random_ports(n: int = 4) -> list[int]:
    """Get N random available ports."""
    import socket

    ports = []
    socks = []
    for _ in range(n):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


class MockSimServer:
    """Minimal mock server for testing SimClient."""

    def __init__(self, ports: list[int]):
        self._ports = ports
        self._ctx = zmq.Context()
        self._running = False
        self._thread: threading.Thread | None = None

        # Ch1: PUB telemetry
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://127.0.0.1:{ports[0]}")

        # Ch2: SUB hints (bind)
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.bind(f"tcp://127.0.0.1:{ports[1]}")
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")

        # Ch3: REP commands
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(f"tcp://127.0.0.1:{ports[2]}")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        for sock in (self._pub, self._sub, self._rep):
            sock.close(linger=0)
        self._ctx.term()

    def _loop(self):
        poller = zmq.Poller()
        poller.register(self._rep, zmq.POLLIN)

        while self._running:
            # Publish telemetry every 50ms
            self._publish_telemetry()

            # Handle commands
            socks = dict(poller.poll(timeout=50))
            if self._rep in socks:
                raw = self._rep.recv()
                msg = msgpack.unpackb(raw, raw=False)
                reply = self._handle_command(msg)
                self._rep.send(msgpack.packb(reply, use_bin_type=True))

    def _publish_telemetry(self):
        msg = pack_telemetry(
            ts_ms=int(time.monotonic() * 1000),
            step_count=0,
            phase_id=0,
            done=False,
            qpos=np.zeros(13),
            qvel=np.zeros(12),
            ee_pose=np.zeros(7),
            object_pose=np.zeros(7),
            joint_pos=np.zeros(6),
            gripper=0.0,
            action=np.zeros(6),
            rgb_scene=np.zeros((48, 64, 3), dtype=np.uint8),
            rgb_wrist=np.zeros((24, 32, 3), dtype=np.uint8),
            jpeg_quality=50,
        )
        packed = msgpack.packb(msg, use_bin_type=True)
        self._pub.send(packed, zmq.NOBLOCK)

    def _handle_command(self, msg: dict) -> dict:
        cmd = msg.get("type")
        if cmd == CMD_STEP:
            return {"type": RESP_STEP_OK, "reward": 0.0, "done": False}
        if cmd == CMD_RESET:
            return {"type": RESP_RESET_OK, "seed": msg.get("seed")}
        if cmd == "shutdown":
            self._running = False
            return {"type": RESP_OK}
        if cmd == "configure":
            return {"type": RESP_OK}
        return {"type": "error", "message": f"Unknown: {cmd}"}


@pytest.fixture
def mock_server():
    ports = _random_ports(4)
    server = MockSimServer(ports)
    server.start()
    time.sleep(0.1)  # let sockets bind
    yield server, ports
    server.stop()


@pytest.fixture
def client_config(mock_server):
    _, ports = mock_server
    return SimBridgeConfig(
        telemetry_url=f"tcp://127.0.0.1:{ports[0]}",
        hints_url=f"tcp://127.0.0.1:{ports[1]}",
        command_url=f"tcp://127.0.0.1:{ports[2]}",
        query_url=f"tcp://127.0.0.1:{ports[3]}",
        recv_timeout_ms=2000,
        command_timeout_ms=2000,
    )


class TestSimClient:
    def test_start_stop(self, mock_server, client_config):
        client = SimClient(client_config)
        client.start(timeout=5.0)
        assert client.started
        assert client.latest_telemetry is not None
        client.stop()
        assert not client.started

    def test_telemetry_reception(self, mock_server, client_config):
        client = SimClient(client_config)
        client.start(timeout=5.0)
        try:
            t = client.latest_telemetry
            assert t is not None
            assert "qpos" in t
            assert t["qpos"].shape == (13,)
            assert t["rgb_scene"].shape[2] == 3
        finally:
            client.stop()

    def test_step_command(self, mock_server, client_config):
        client = SimClient(client_config)
        client.start(timeout=5.0)
        try:
            resp = client.step(np.zeros(6))
            assert resp["type"] == RESP_STEP_OK
            assert resp["done"] is False
        finally:
            client.stop()

    def test_reset_command(self, mock_server, client_config):
        client = SimClient(client_config)
        client.start(timeout=5.0)
        try:
            resp = client.reset(seed=42)
            assert resp["type"] == RESP_RESET_OK
            assert resp["seed"] == 42
        finally:
            client.stop()

    def test_configure_command(self, mock_server, client_config):
        client = SimClient(client_config)
        client.start(timeout=5.0)
        try:
            resp = client.configure(teacher_mode=True)
            assert resp["type"] == RESP_OK
        finally:
            client.stop()

    def test_not_started_raises(self):
        client = SimClient(SimBridgeConfig())
        with pytest.raises(BridgeTransportError, match="not started"):
            client.step(np.zeros(6))

    def test_double_start_is_noop(self, mock_server, client_config):
        client = SimClient(client_config)
        client.start(timeout=5.0)
        try:
            client.start(timeout=5.0)  # should not raise
            assert client.started
        finally:
            client.stop()

    def test_publish_hint(self, mock_server, client_config):
        client = SimClient(client_config)
        client.start(timeout=5.0)
        try:
            # Should not raise
            client.publish_hint(
                target_handle="cube",
                bbox_xywh=(100, 100, 50, 50),
                confidence=0.95,
                tracker_ok=True,
            )
        finally:
            client.stop()

    def test_parse_port(self):
        assert SimClient._parse_port("tcp://127.0.0.1:5560") == 5560
        assert SimClient._parse_port("tcp://10.0.0.1:7000") == 7000
