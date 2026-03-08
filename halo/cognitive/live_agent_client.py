"""LiveAgentClient — TUI-side WebSocket client for the Live Agent.

Connects to the cloud service's ``/ws/live/{arm_id}`` endpoint and provides:
- ``send_audio(pcm_bytes)`` — forward mic PCM to the Live Agent
- ``send_text(msg)`` — send text input to the Live Agent
- ``send_event(event_dict)`` — forward robot events for narration
- Receives audio, text, transcriptions, status, interrupt, and commands
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MAX_RECONNECT_DELAY_S = 30.0


@dataclass
class LiveAgentClientState:
    """Observable state of the Live Agent client for TUI display."""

    connected: bool = False
    last_transcription_in: str = ""
    last_transcription_out: str = ""
    last_text_out: str = ""
    last_status: str = ""
    reconnect_count: int = 0


class LiveAgentClient:
    """WebSocket client that connects to the cloud service Live Agent endpoint.

    Audio flows bidirectionally: mic PCM → cloud → Gemini → audio → speaker.
    Text flows: operator text → cloud → Gemini → text/audio response.
    Commands flow back from the planner via the Live Agent.
    """

    def __init__(
        self,
        url: str,
        arm_id: str,
        on_audio_out: object | None = None,
        on_interrupt: object | None = None,
        on_tool_call: object | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._arm_id = arm_id
        self._ws_url = f"{self._url}/ws/live/{arm_id}"
        self._on_audio_out = on_audio_out  # Callable[[bytes], None]
        self._on_interrupt = on_interrupt  # Callable[[], None]
        self._on_tool_call = on_tool_call  # Callable[[str, str, dict], Awaitable[None] | None]

        self._state = LiveAgentClientState()
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._audio_send_count = 0

        # Pending commands from the Live Agent
        self._pending_commands: asyncio.Queue[list[dict]] = asyncio.Queue()

    @property
    def state(self) -> LiveAgentClientState:
        return self._state

    async def connect(self) -> None:
        """Connect to the Live Agent WebSocket with reconnection."""
        if self._started:
            return
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._recv_task = asyncio.create_task(self._connection_loop())

    async def disconnect(self) -> None:
        """Disconnect from the Live Agent WebSocket."""
        self._started = False
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._state.connected = False

    def send_audio(self, pcm_bytes: bytes) -> None:
        """Send PCM audio to the Live Agent (fire-and-forget).

        Called from sounddevice's audio callback thread, so we must use
        run_coroutine_threadsafe instead of ensure_future.
        """
        if self._ws is None or not self._state.connected or self._loop is None:
            if self._audio_send_count == 0:
                logger.debug(
                    "send_audio: dropping (ws=%s, connected=%s, loop=%s)",
                    self._ws is not None,
                    self._state.connected,
                    self._loop is not None,
                )
            return
        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        msg = json.dumps({"type": "audio_in", "data": encoded})
        asyncio.run_coroutine_threadsafe(self._ws_send(msg), self._loop)
        self._audio_send_count += 1
        if self._audio_send_count == 1:
            logger.info("send_audio: first audio chunk sent (%d bytes)", len(pcm_bytes))
        elif self._audio_send_count % 100 == 0:
            logger.debug("send_audio: %d chunks sent", self._audio_send_count)

    async def send_text(self, text: str) -> None:
        """Send text input to the Live Agent."""
        if self._ws is None or not self._state.connected:
            return
        msg = json.dumps({"type": "text_in", "text": text})
        await self._ws_send(msg)

    async def send_event(self, event_dict: dict) -> None:
        """Send a robot event to the Live Agent for narration."""
        if self._ws is None or not self._state.connected:
            return
        msg = json.dumps({"type": "event", "data": event_dict})
        await self._ws_send(msg)

    async def send_tool_result(self, call_id: str, result: str) -> None:
        """Send a tool result back to the cloud Live Agent."""
        if self._ws is None or not self._state.connected:
            return
        msg = json.dumps({"type": "tool_result", "call_id": call_id, "result": result})
        await self._ws_send(msg)

    def drain_commands(self) -> list[list[dict]]:
        """Drain any pending command batches from the Live Agent."""
        batches = []
        while not self._pending_commands.empty():
            try:
                batches.append(self._pending_commands.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batches

    def set_audio_callbacks(
        self,
        on_audio_out: object | None = None,
        on_interrupt: object | None = None,
    ) -> None:
        """Set or update audio callbacks (e.g. after creating AudioPlayback)."""
        self._on_audio_out = on_audio_out
        self._on_interrupt = on_interrupt

    async def aclose(self) -> None:
        """Alias for disconnect()."""
        await self.disconnect()

    async def _ws_send(self, msg: str) -> None:
        """Send a message, ignoring errors from closed connections."""
        try:
            if self._ws is not None:
                await self._ws.send(msg)
        except Exception:
            pass

    async def _connection_loop(self) -> None:
        """Reconnection loop with exponential backoff."""
        delay = 1.0
        while self._started:
            try:
                await self._run_session()
                delay = 1.0  # Reset on clean exit
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Live agent WS connection failed, reconnecting in %.1fs", delay)
                self._state.connected = False
                self._state.reconnect_count += 1
                await asyncio.sleep(delay)
                delay = min(delay * 2, _MAX_RECONNECT_DELAY_S)

    async def _run_session(self) -> None:
        """Single WebSocket session — receive messages until disconnect."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package required for Live Agent client. Install with: pip install websockets")
            raise

        # Convert http(s) URL to ws(s) for WebSocket connection
        ws_url = self._ws_url
        if ws_url.startswith("http://"):
            ws_url = "ws://" + ws_url[7:]
        elif ws_url.startswith("https://"):
            ws_url = "wss://" + ws_url[8:]

        async with websockets.connect(ws_url) as ws:
            self._ws = ws
            self._state.connected = True
            logger.info("Live agent WS connected to %s", ws_url)

            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")
                    self._handle_message(msg_type, msg)
            finally:
                self._ws = None
                self._state.connected = False

    def _handle_message(self, msg_type: str | None, msg: dict) -> None:
        """Process an inbound WebSocket message."""
        if msg_type == "audio_out":
            data = msg.get("data", "")
            try:
                pcm = base64.b64decode(data)
                if self._on_audio_out and callable(self._on_audio_out):
                    self._on_audio_out(pcm)
            except Exception:
                pass

        elif msg_type == "text_out":
            text = msg.get("text", "")
            self._state.last_text_out = text

        elif msg_type == "transcription_in":
            self._state.last_transcription_in = msg.get("text", "")

        elif msg_type == "transcription_out":
            self._state.last_transcription_out = msg.get("text", "")

        elif msg_type == "status":
            self._state.last_status = msg.get("text", "")

        elif msg_type == "interrupt":
            if self._on_interrupt and callable(self._on_interrupt):
                self._on_interrupt()

        elif msg_type == "tool_call":
            call_id = msg.get("call_id", "")
            name = msg.get("name", "")
            args = msg.get("args", {})
            if self._on_tool_call and callable(self._on_tool_call):
                # Tool handlers may be async — schedule as task
                result = self._on_tool_call(call_id, name, args)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)

        elif msg_type == "commands":
            commands = msg.get("data", [])
            if commands:
                self._pending_commands.put_nowait(commands)
