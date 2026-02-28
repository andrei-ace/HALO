"""Tests for the MuJoCo sim server protocol and command handling.

Tests protocol encoding/decoding, command dispatch, and server config.
Does NOT require a running server — tests components in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Protocol tests (no MuJoCo dependency)
# ---------------------------------------------------------------------------


class TestProtocol:
    """Tests for protocol.py encode/decode helpers."""

    def test_ndarray_roundtrip(self):
        from mujoco_sim.server.protocol import bytes_to_ndarray, ndarray_to_bytes

        arr = np.array([1.0, 2.0, 3.0, 4.5, -1.7], dtype=np.float64)
        buf = ndarray_to_bytes(arr)
        assert isinstance(buf, bytes)

        recovered = bytes_to_ndarray(buf, shape=(5,))
        np.testing.assert_array_almost_equal(recovered, arr)

    def test_ndarray_roundtrip_2d(self):
        from mujoco_sim.server.protocol import bytes_to_ndarray, ndarray_to_bytes

        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        buf = ndarray_to_bytes(arr)
        recovered = bytes_to_ndarray(buf, shape=(3, 4))
        np.testing.assert_array_equal(recovered, arr)

    def test_ndarray_writable(self):
        """Recovered arrays must be writable (not read-only views)."""
        from mujoco_sim.server.protocol import bytes_to_ndarray, ndarray_to_bytes

        arr = np.array([1.0, 2.0], dtype=np.float64)
        buf = ndarray_to_bytes(arr)
        recovered = bytes_to_ndarray(buf, shape=(2,))
        recovered[0] = 99.0  # must not raise
        assert recovered[0] == 99.0

    def test_jpeg_roundtrip(self):
        from mujoco_sim.server.protocol import jpeg_decode, jpeg_encode

        # Use a smooth gradient (not random noise) — JPEG handles real images well
        h, w = 48, 64
        r = np.tile(np.linspace(0, 200, w, dtype=np.uint8), (h, 1))
        g = np.tile(np.linspace(50, 150, h, dtype=np.uint8).reshape(-1, 1), (1, w))
        b = np.full((h, w), 100, dtype=np.uint8)
        rgb = np.stack([r, g, b], axis=-1)

        buf = jpeg_encode(rgb, quality=95)
        assert isinstance(buf, bytes)
        assert len(buf) < rgb.nbytes  # compressed

        recovered = jpeg_decode(buf)
        assert recovered.shape == rgb.shape
        assert recovered.dtype == np.uint8
        # JPEG is lossy — check approximate match
        diff = np.abs(recovered.astype(float) - rgb.astype(float))
        assert diff.mean() < 5.0

    def test_jpeg_encode_quality(self):
        from mujoco_sim.server.protocol import jpeg_encode

        rgb = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        low_q = jpeg_encode(rgb, quality=10)
        high_q = jpeg_encode(rgb, quality=95)
        assert len(low_q) < len(high_q)

    def test_pack_unpack_telemetry(self):
        from mujoco_sim.server.protocol import pack_telemetry, unpack_telemetry

        qpos = np.random.randn(13)
        qvel = np.random.randn(12)
        ee_pose = np.random.randn(7)
        object_pose = np.random.randn(7)
        joint_pos = np.random.randn(6)
        action = np.random.randn(6)
        rgb_scene = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        rgb_wrist = np.random.randint(0, 255, (24, 32, 3), dtype=np.uint8)

        msg = pack_telemetry(
            ts_ms=12345,
            step_count=42,
            phase_id=5,
            done=False,
            qpos=qpos,
            qvel=qvel,
            ee_pose=ee_pose,
            object_pose=object_pose,
            joint_pos=joint_pos,
            gripper=-0.17,
            action=action,
            rgb_scene=rgb_scene,
            rgb_wrist=rgb_wrist,
            jpeg_quality=95,
        )
        assert msg["type"] == "telemetry"
        assert isinstance(msg["qpos"], bytes)
        assert isinstance(msg["rgb_scene"], bytes)

        decoded = unpack_telemetry(msg)
        assert decoded["ts_ms"] == 12345
        assert decoded["step_count"] == 42
        assert decoded["phase_id"] == 5
        assert decoded["done"] is False
        assert decoded["gripper"] == pytest.approx(-0.17)
        np.testing.assert_array_almost_equal(decoded["qpos"], qpos)
        np.testing.assert_array_almost_equal(decoded["qvel"], qvel)
        np.testing.assert_array_almost_equal(decoded["ee_pose"], ee_pose)
        np.testing.assert_array_almost_equal(decoded["joint_pos"], joint_pos)
        np.testing.assert_array_almost_equal(decoded["action"], action)
        assert decoded["rgb_scene"].shape == (48, 64, 3)
        assert decoded["rgb_wrist"].shape == (24, 32, 3)

    def test_message_type_constants(self):
        from mujoco_sim.server.protocol import (
            CMD_CONFIGURE,
            CMD_GET_STATE,
            CMD_RESET,
            CMD_SET_STATE,
            CMD_SHUTDOWN,
            CMD_STEP,
            CMD_TEACHER_STEP,
            MSG_TELEMETRY,
            MSG_TRACKING_HINT,
            QUERY_TRACKER_INIT,
            QUERY_TRACKER_UPDATE,
            QUERY_VLM_DETECT,
            RESP_ERROR,
            RESP_OK,
            RESP_RESET_OK,
            RESP_STATE,
            RESP_STEP_OK,
            RESP_TEACHER_STEP_OK,
        )

        # Just verify they're strings and distinct
        all_types = [
            MSG_TELEMETRY,
            MSG_TRACKING_HINT,
            CMD_STEP,
            CMD_RESET,
            CMD_GET_STATE,
            CMD_SET_STATE,
            CMD_TEACHER_STEP,
            CMD_CONFIGURE,
            CMD_SHUTDOWN,
            RESP_STEP_OK,
            RESP_RESET_OK,
            RESP_STATE,
            RESP_TEACHER_STEP_OK,
            RESP_OK,
            RESP_ERROR,
            QUERY_VLM_DETECT,
            QUERY_TRACKER_INIT,
            QUERY_TRACKER_UPDATE,
        ]
        assert all(isinstance(t, str) for t in all_types)
        assert len(set(all_types)) == len(all_types)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestSimServerConfig:
    def test_defaults(self):
        from mujoco_sim.server.config import SimServerConfig

        cfg = SimServerConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.telemetry_port == 5560
        assert cfg.hints_port == 5561
        assert cfg.command_port == 5562
        assert cfg.query_port == 5563
        assert cfg.render_fps == 10
        assert cfg.jpeg_quality == 85

    def test_urls(self):
        from mujoco_sim.server.config import SimServerConfig

        cfg = SimServerConfig(host="192.168.1.1", telemetry_port=6000)
        assert cfg.telemetry_url == "tcp://192.168.1.1:6000"
        assert cfg.command_url == "tcp://192.168.1.1:5562"


# ---------------------------------------------------------------------------
# Command handler tests (require MuJoCo for env/teacher)
# ---------------------------------------------------------------------------

try:
    import mujoco as _mj  # noqa: F401

    _has_mujoco = True
except ImportError:
    _has_mujoco = False

skip_no_mujoco = pytest.mark.skipif(not _has_mujoco, reason="mujoco not installed")


@skip_no_mujoco
class TestHandlers:
    """Tests for handlers.py command dispatch."""

    @pytest.fixture
    def env_teacher(self):
        from mujoco_sim.config import EnvConfig
        from mujoco_sim.env import SO101Env
        from mujoco_sim.teacher.pick_teacher import PickTeacher, TeacherConfig

        env = SO101Env(EnvConfig())
        teacher = PickTeacher(TeacherConfig())
        env.reset(seed=42)
        yield env, teacher
        env.close()

    def test_step(self, env_teacher):
        from mujoco_sim.server.handlers import dispatch_command
        from mujoco_sim.server.protocol import ndarray_to_bytes

        env, teacher = env_teacher
        action = np.zeros(6)
        msg = {"type": "step", "action": ndarray_to_bytes(action)}
        reply, shutdown = dispatch_command(msg, env, teacher)
        assert reply["type"] == "step_ok"
        assert isinstance(reply["reward"], float)
        assert isinstance(reply["done"], bool)
        assert shutdown is False

    def test_reset(self, env_teacher):
        from mujoco_sim.server.handlers import dispatch_command

        env, teacher = env_teacher
        msg = {"type": "reset", "seed": 99}
        reply, shutdown = dispatch_command(msg, env, teacher)
        assert reply["type"] == "reset_ok"
        assert reply["seed"] == 99
        assert shutdown is False

    def test_get_set_state(self, env_teacher):
        from mujoco_sim.server.handlers import dispatch_command
        from mujoco_sim.server.protocol import ndarray_to_bytes

        env, teacher = env_teacher
        # Get state
        reply, _ = dispatch_command({"type": "get_state"}, env, teacher)
        assert reply["type"] == "state"
        assert isinstance(reply["qpos"], bytes)
        assert isinstance(reply["qvel"], bytes)

        # Set state
        qpos = np.zeros(13)
        qvel = np.zeros(12)
        msg = {"type": "set_state", "qpos": ndarray_to_bytes(qpos), "qvel": ndarray_to_bytes(qvel)}
        reply, _ = dispatch_command(msg, env, teacher)
        assert reply["type"] == "ok"

    def test_teacher_step_requires_mode(self, env_teacher):
        from mujoco_sim.server.handlers import dispatch_command

        env, teacher = env_teacher
        reply, _ = dispatch_command({"type": "teacher_step"}, env, teacher, teacher_mode=False)
        assert reply["type"] == "error"
        assert "teacher_mode" in reply["message"]

    def test_teacher_step_with_mode(self, env_teacher):
        from mujoco_sim.server.handlers import dispatch_command

        env, teacher = env_teacher
        reply, _ = dispatch_command({"type": "teacher_step"}, env, teacher, teacher_mode=True)
        assert reply["type"] == "teacher_step_ok"
        assert isinstance(reply["action"], bytes)
        assert isinstance(reply["phase_id"], int)
        assert isinstance(reply["done"], bool)

    def test_shutdown(self, env_teacher):
        from mujoco_sim.server.handlers import dispatch_command

        env, teacher = env_teacher
        reply, shutdown = dispatch_command({"type": "shutdown"}, env, teacher)
        assert reply["type"] == "ok"
        assert shutdown is True

    def test_unknown_command(self, env_teacher):
        from mujoco_sim.server.handlers import dispatch_command

        env, teacher = env_teacher
        reply, shutdown = dispatch_command({"type": "bogus"}, env, teacher)
        assert reply["type"] == "error"
        assert shutdown is False

    def test_configure(self, env_teacher):
        from mujoco_sim.server.handlers import dispatch_command

        env, teacher = env_teacher
        msg = {"type": "configure", "teacher_mode": True}
        reply, shutdown = dispatch_command(msg, env, teacher)
        assert reply["type"] == "ok"
        assert shutdown is False
