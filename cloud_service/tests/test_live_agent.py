"""Unit tests for the Live Agent proxy-tool architecture."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Proxy tool callback & resolve ──


class TestProxyToolCallback:
    @pytest.mark.asyncio
    async def test_proxy_callback_returns_result(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")

        # Track tool calls via callback
        tool_calls = []

        def on_tool_call(call_id, name, args):
            tool_calls.append((call_id, name, args))
            # Simulate TUI responding immediately
            session.resolve_tool_call(call_id, "robot is idle")

        session.set_callbacks(on_tool_call=on_tool_call)

        # Simulate ADK calling before_tool_callback
        mock_tool = MagicMock()
        mock_tool.name = "get_robot_state"
        mock_context = MagicMock()

        result = await session._proxy_before_tool_callback(mock_tool, {}, mock_context)

        assert result == {"result": "robot is idle"}
        assert len(tool_calls) == 1
        assert tool_calls[0][1] == "get_robot_state"

    @pytest.mark.asyncio
    async def test_proxy_callback_with_args(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")

        def on_tool_call(call_id, name, args):
            session.resolve_tool_call(call_id, f"scene: {args.get('reason', '')}")

        session.set_callbacks(on_tool_call=on_tool_call)

        mock_tool = MagicMock()
        mock_tool.name = "describe_scene"
        mock_context = MagicMock()

        result = await session._proxy_before_tool_callback(mock_tool, {"reason": "user asked"}, mock_context)

        assert result == {"result": "scene: user asked"}

    @pytest.mark.asyncio
    async def test_proxy_callback_timeout(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")
        # No on_tool_call callback — nobody resolves the future

        mock_tool = MagicMock()
        mock_tool.name = "get_robot_state"
        mock_context = MagicMock()

        # Patch timeout to be very short
        with patch("cloud_service.live_agent._TOOL_TIMEOUT_S", 0.05):
            result = await session._proxy_before_tool_callback(mock_tool, {}, mock_context)

        assert "timed out" in result["result"]
        # Future should be cleaned up
        assert len(session._tool_results) == 0

    @pytest.mark.asyncio
    async def test_proxy_callback_no_callback_set(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")
        # No on_tool_call set — should still work (timeout)

        mock_tool = MagicMock()
        mock_tool.name = "submit_user_intent"
        mock_context = MagicMock()

        with patch("cloud_service.live_agent._TOOL_TIMEOUT_S", 0.05):
            result = await session._proxy_before_tool_callback(mock_tool, {"intent": "pick cube"}, mock_context)

        assert "timed out" in result["result"]

    @pytest.mark.asyncio
    async def test_resolve_unknown_call_id(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")
        # Should not raise
        session.resolve_tool_call("nonexistent", "result")

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")

        pending = {}

        def on_tool_call(call_id, name, args):
            pending[call_id] = name

        session.set_callbacks(on_tool_call=on_tool_call)

        mock_context = MagicMock()

        # Launch two tool calls concurrently
        tool1 = MagicMock()
        tool1.name = "get_robot_state"
        tool2 = MagicMock()
        tool2.name = "describe_scene"

        async def call1():
            return await session._proxy_before_tool_callback(tool1, {}, mock_context)

        async def call2():
            return await session._proxy_before_tool_callback(tool2, {"reason": "test"}, mock_context)

        task1 = asyncio.create_task(call1())
        task2 = asyncio.create_task(call2())

        # Wait a bit for callbacks to fire
        await asyncio.sleep(0.01)

        # Resolve both
        for call_id, name in list(pending.items()):
            session.resolve_tool_call(call_id, f"{name} result")

        r1 = await task1
        r2 = await task2

        assert r1 == {"result": "get_robot_state result"}
        assert r2 == {"result": "describe_scene result"}


# ── LiveAgentSession ──


class TestLiveAgentSession:
    def test_initial_state(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")
        assert session.arm_id == "arm0"
        assert session.state.connected is False

    def test_set_callbacks(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")
        cb = MagicMock()
        tool_cb = MagicMock()
        session.set_callbacks(on_text_out=cb, on_tool_call=tool_cb)
        assert session._on_text_out is cb
        assert session._on_tool_call is tool_cb

    def test_inject_status_throttle(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")
        session._queue = MagicMock()
        session._state.connected = True
        session.inject_status("first")
        first_ts = session._last_status_inject_ts
        assert first_ts > 0

        # Second call within throttle window should be skipped
        session.inject_status("second")
        assert session._last_status_inject_ts == first_ts

    def test_send_text_when_not_connected(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")
        session.send_text("hello")

    def test_on_audio_chunk_when_not_connected(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")
        session.on_audio_chunk(b"\x00" * 100)

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_futures(self):
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session._tool_results["test-id"] = future
        session._started = True

        await session.stop()

        assert future.cancelled()
        assert len(session._tool_results) == 0


# ── LiveAgentManager ──


class TestLiveAgentManager:
    @pytest.mark.asyncio
    async def test_create_and_get(self):
        from cloud_service.live_agent_manager import LiveAgentManager

        mgr = LiveAgentManager()

        with patch("cloud_service.live_agent_manager.LiveAgentSession") as MockSession:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            MockSession.return_value = mock_instance

            session = await mgr.get_or_create("arm0")
            assert session is mock_instance
            mock_instance.start.assert_awaited_once()

            # Second call returns same session
            session2 = await mgr.get_or_create("arm0")
            assert session2 is mock_instance

    @pytest.mark.asyncio
    async def test_remove(self):
        from cloud_service.live_agent_manager import LiveAgentManager

        mgr = LiveAgentManager()

        with patch("cloud_service.live_agent_manager.LiveAgentSession") as MockSession:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            MockSession.return_value = mock_instance

            await mgr.get_or_create("arm0")
            await mgr.remove("arm0")
            mock_instance.stop.assert_awaited_once()
            assert "arm0" not in mgr.active_arm_ids

    @pytest.mark.asyncio
    async def test_remove_all(self):
        from cloud_service.live_agent_manager import LiveAgentManager

        mgr = LiveAgentManager()

        with patch("cloud_service.live_agent_manager.LiveAgentSession") as MockSession:
            instances = []
            for _ in range(3):
                inst = MagicMock()
                inst.start = AsyncMock()
                inst.stop = AsyncMock()
                instances.append(inst)
            MockSession.side_effect = instances

            await mgr.get_or_create("arm0")
            await mgr.get_or_create("arm1")
            await mgr.get_or_create("arm2")

            await mgr.remove_all()
            assert len(mgr.active_arm_ids) == 0

    @pytest.mark.asyncio
    async def test_no_session_mgr_required(self):
        """Manager no longer requires session_mgr or snapshot_cache."""
        from cloud_service.live_agent_manager import LiveAgentManager

        mgr = LiveAgentManager(model="test-model", voice="TestVoice")
        assert mgr._model == "test-model"
        assert mgr._voice == "TestVoice"


# ── WebSocket handler ──


class TestWSHandler:
    @pytest.mark.asyncio
    async def test_ws_endpoint_not_enabled(self):
        """WS connection closes with 1011 when Live Agent is not enabled."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from cloud_service.ws_handler import router

        test_app = FastAPI()
        test_app.include_router(router)
        # No live_agent_manager on app.state

        client = TestClient(test_app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/live/arm0"):
                pass

    @pytest.mark.asyncio
    async def test_tool_result_message_resolves_call(self):
        """Verify that a tool_result WS message calls resolve_tool_call on session."""
        from cloud_service.live_agent import LiveAgentSession

        session = LiveAgentSession(arm_id="arm0")

        # Set up a pending future
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session._tool_results["call-123"] = future

        session.resolve_tool_call("call-123", "the result")

        assert future.result() == "the result"


# ── LiveAgentClient ──


class TestLiveAgentClient:
    def test_initial_state(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0")
        assert client.state.connected is False
        assert client._ws_url == "http://localhost:8000/ws/live/arm0"

    def test_send_audio_when_disconnected(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0")
        client.send_audio(b"\x00" * 100)

    def test_drain_commands_empty(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0")
        assert client.drain_commands() == []

    def test_handle_audio_out(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        received = []
        client = LiveAgentClient(
            url="http://localhost:8000",
            arm_id="arm0",
            on_audio_out=lambda pcm: received.append(pcm),
        )

        pcm = b"\x01\x02\x03\x04"
        encoded = base64.b64encode(pcm).decode("ascii")
        client._handle_message("audio_out", {"type": "audio_out", "data": encoded})
        assert received == [pcm]

    def test_handle_text_out(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0")
        client._handle_message("text_out", {"type": "text_out", "text": "Hello operator"})
        assert client.state.last_text_out == "Hello operator"

    def test_handle_transcription(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0")
        client._handle_message("transcription_in", {"type": "transcription_in", "text": "pick the cube"})
        assert client.state.last_transcription_in == "pick the cube"

        client._handle_message("transcription_out", {"type": "transcription_out", "text": "Starting pick"})
        assert client.state.last_transcription_out == "Starting pick"

    def test_handle_status(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0")
        client._handle_message("status", {"type": "status", "text": "Pick succeeded"})
        assert client.state.last_status == "Pick succeeded"

    def test_handle_interrupt(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        interrupted = []
        client = LiveAgentClient(
            url="http://localhost:8000",
            arm_id="arm0",
            on_interrupt=lambda: interrupted.append(True),
        )
        client._handle_message("interrupt", {"type": "interrupt"})
        assert interrupted == [True]

    def test_handle_commands(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0")
        commands = [{"type": "START_SKILL", "payload": {"skill_name": "PICK"}}]
        client._handle_message("commands", {"type": "commands", "data": commands})
        batches = client.drain_commands()
        assert len(batches) == 1
        assert batches[0] == commands

    def test_handle_tool_call(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        tool_calls = []
        client = LiveAgentClient(
            url="http://localhost:8000",
            arm_id="arm0",
            on_tool_call=lambda call_id, name, args: tool_calls.append((call_id, name, args)),
        )
        client._handle_message(
            "tool_call",
            {"type": "tool_call", "call_id": "abc-123", "name": "get_robot_state", "args": {}},
        )
        assert len(tool_calls) == 1
        assert tool_calls[0] == ("abc-123", "get_robot_state", {})

    def test_handle_tool_call_with_args(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        tool_calls = []
        client = LiveAgentClient(
            url="http://localhost:8000",
            arm_id="arm0",
            on_tool_call=lambda call_id, name, args: tool_calls.append((call_id, name, args)),
        )
        client._handle_message(
            "tool_call",
            {
                "type": "tool_call",
                "call_id": "def-456",
                "name": "describe_scene",
                "args": {"reason": "user asked"},
            },
        )
        assert tool_calls[0] == ("def-456", "describe_scene", {"reason": "user asked"})

    def test_handle_tool_call_no_callback(self):
        """tool_call message with no on_tool_call callback should not raise."""
        from halo.cognitive.live_agent_client import LiveAgentClient

        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0")
        # Should not raise
        client._handle_message(
            "tool_call",
            {"type": "tool_call", "call_id": "xyz", "name": "get_robot_state", "args": {}},
        )

    def test_set_audio_callbacks(self):
        from halo.cognitive.live_agent_client import LiveAgentClient

        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0")
        assert client._on_audio_out is None
        assert client._on_interrupt is None

        audio_out = MagicMock()
        interrupt = MagicMock()
        client.set_audio_callbacks(on_audio_out=audio_out, on_interrupt=interrupt)

        assert client._on_audio_out is audio_out
        assert client._on_interrupt is interrupt

        # Verify callbacks are invoked on message handling
        pcm = b"\x01\x02"
        encoded = base64.b64encode(pcm).decode("ascii")
        client._handle_message("audio_out", {"type": "audio_out", "data": encoded})
        audio_out.assert_called_once_with(pcm)

        client._handle_message("interrupt", {"type": "interrupt"})
        interrupt.assert_called_once()

    def test_set_audio_callbacks_clear(self):
        """Setting callbacks to None clears them."""
        from halo.cognitive.live_agent_client import LiveAgentClient

        audio_out = MagicMock()
        client = LiveAgentClient(url="http://localhost:8000", arm_id="arm0", on_audio_out=audio_out)
        assert client._on_audio_out is audio_out

        client.set_audio_callbacks()  # clear
        assert client._on_audio_out is None
        assert client._on_interrupt is None


# ── Config ──


class TestConfig:
    def test_live_agent_config_defaults(self):
        from halo.cognitive.config import LiveAgentConfig

        cfg = LiveAgentConfig()
        assert cfg.enabled is False
        assert cfg.voice_name == "Kore"
        assert cfg.model == "gemini-2.5-flash-native-audio-preview-12-2025"

    def test_service_config_live_agent_fields(self):
        from cloud_service.config import ServiceConfig

        cfg = ServiceConfig()
        assert cfg.live_agent_enabled is True
        assert cfg.live_agent_model == "gemini-2.5-flash-native-audio-preview-12-2025"
        assert cfg.live_agent_voice == "Kore"

    def test_service_config_from_env(self):
        import os

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test", "HALO_LIVE_AGENT_ENABLED": "false"}):
            from cloud_service.config import ServiceConfig

            cfg = ServiceConfig.from_env()
            assert cfg.live_agent_enabled is False
