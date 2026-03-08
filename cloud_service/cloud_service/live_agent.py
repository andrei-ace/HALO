"""LiveAgentSession — conversational voice/text agent via Gemini Live bidi streaming.

Sits between the operator and the planner. Interprets natural language,
answers questions about robot state, and forwards structured intent to the
planner via ``submit_user_intent``. Never issues robot commands directly.

Uses a **proxy-tool architecture**: tool functions are dummy stubs (bodies never
execute). ADK's ``before_tool_callback`` intercepts every tool call, serializes
it over WebSocket to the TUI, the TUI executes locally with fresh runtime state,
and sends the result back. The cloud feeds the result to Gemini.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Callable

from google.adk.agents import Agent, LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

logger = logging.getLogger(__name__)

_APP_NAME = "halo-live_agent"
_USER_ID = "operator"
_MAX_RECONNECT_DELAY_S = 30.0
_TOOL_TIMEOUT_S = 30.0
_MONITOR_QUEUE_SIZE = 64


class LiveAgentState:
    """Observable state of the live agent session for TUI display."""

    __slots__ = (
        "connected",
        "last_transcription_in",
        "last_transcription_out",
        "last_text_out",
        "turn_active",
        "reconnect_count",
    )

    def __init__(self) -> None:
        self.connected: bool = False
        self.last_transcription_in: str = ""
        self.last_transcription_out: str = ""
        self.last_text_out: str = ""
        self.turn_active: bool = False
        self.reconnect_count: int = 0


def _load_system_prompt(prompts_dir: Path) -> str:
    """Load the live agent system prompt."""
    prompt_path = prompts_dir / "system_prompt.md"
    return prompt_path.read_text(encoding="utf-8")


# ── Dummy tool functions ──
# Bodies never execute — before_tool_callback always intercepts.
# Docstrings are used by Gemini for function-calling schema generation.


async def describe_scene(reason: str = "") -> str:
    """Ask the vision system to describe the current scene. Returns a text description of visible objects.

    Args:
        reason: Why the scene description is needed (e.g., "user asked what's on the table").
    """
    return ""


async def submit_user_intent(intent: str) -> str:
    """Forward a user instruction to the robot planner for execution.

    The planner will interpret the intent and issue appropriate robot commands.
    Use this for action requests like pick, place, move, etc. Do NOT use for abort/stop.

    Args:
        intent: The operator's instruction in natural language (e.g., "pick up the red cube").
    """
    return ""


async def abort() -> str:
    """Immediately abort the current skill and clear all queued skills.

    Use this when the operator says stop, abort, cancel, or halt.
    This is faster than submit_user_intent — it bypasses the planner and acts directly.
    """
    return ""


_LIVE_AGENT_TOOLS = [describe_scene, submit_user_intent, abort]


class LiveAgentSession:
    """Persistent bidirectional streaming session to Gemini via ADK run_live().

    Supports two parallel data flows:
    1. **Audio** (continuous): mic PCM → send_realtime(Blob) → Gemini → audio → speaker.
    2. **Text** (request-response): send_content(Content) → Gemini → text/audio response.

    Tools are proxied to the TUI via WebSocket: when Gemini calls a tool,
    ``before_tool_callback`` serializes the call, sends it to the TUI, awaits
    the result, and returns it to ADK/Gemini.
    """

    def __init__(
        self,
        arm_id: str,
        model: str = "gemini-2.5-flash-native-audio-preview-12-2025",
        voice: str = "Aoede",
        prompts_dir: Path | None = None,
    ) -> None:
        self._arm_id = arm_id
        self._model = model
        self._voice = voice
        self._prompts_dir = prompts_dir or Path(__file__).parents[2] / "configs" / "live_agent"

        self._state = LiveAgentState()
        self._session_id = f"live_agent-{arm_id}"

        # Built lazily in start()
        self._agent: Agent | None = None
        self._runner: Runner | None = None
        self._session_service: InMemorySessionService | None = None
        self._queue: LiveRequestQueue | None = None
        self._event_loop_task: asyncio.Task | None = None
        self._started = False

        # Callbacks registered by the WS handler
        self._on_audio_out: Callable[[bytes], None] | None = None
        self._on_text_out: Callable[[str], None] | None = None
        self._on_transcription_in: Callable[[str, bool], None] | None = None
        self._on_transcription_out: Callable[[str, bool], None] | None = None
        self._on_tool_call: Callable[[str, str, dict], None] | None = None

        # Proxy-tool infrastructure
        self._tool_results: dict[str, asyncio.Future] = {}

        # Monitor queue — streaming tool reads from this
        self._monitor_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=_MONITOR_QUEUE_SIZE)

    @property
    def state(self) -> LiveAgentState:
        return self._state

    @property
    def arm_id(self) -> str:
        return self._arm_id

    def set_callbacks(
        self,
        on_audio_out: Callable[[bytes], None] | None = None,
        on_text_out: Callable[[str], None] | None = None,
        on_transcription_in: Callable[[str, bool], None] | None = None,
        on_transcription_out: Callable[[str, bool], None] | None = None,
        on_tool_call: Callable[[str, str, dict], None] | None = None,
    ) -> None:
        """Register callbacks for outbound data (used by WS handler)."""
        self._on_audio_out = on_audio_out
        self._on_text_out = on_text_out
        self._on_transcription_in = on_transcription_in
        self._on_transcription_out = on_transcription_out
        self._on_tool_call = on_tool_call

    def resolve_tool_call(self, call_id: str, result: str) -> None:
        """Resolve a pending proxy tool call with the TUI-provided result.

        Called by the WS handler when the TUI sends a ``tool_result`` message.
        """
        future = self._tool_results.get(call_id)
        if future is not None and not future.done():
            future.set_result(result)
        else:
            logger.warning("resolve_tool_call: unknown or already-resolved call_id=%s", call_id)

    def put_monitor_update(self, text: str) -> None:
        """Enqueue a monitor update for the streaming ``monitor()`` tool.

        Silently drops the update if the queue is full.
        """
        try:
            self._monitor_queue.put_nowait(text)
        except asyncio.QueueFull:
            pass

    async def start(self) -> None:
        """Create ADK Agent, Runner, session, and start the event loop."""
        if self._started:
            return

        # Suppress Pydantic serialization warnings for response_modalities enum
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

        system_prompt = _load_system_prompt(self._prompts_dir)

        # Build the monitor streaming tool as a closure over our queue
        monitor_queue = self._monitor_queue

        async def monitor() -> AsyncGenerator[str, None]:
            """Stream real-time robot status updates.

            Call this at session start. Yields ``[Event]``, ``[Planner]``,
            and ``[Scene]`` updates as they happen — no polling needed.
            """
            while True:
                item = await monitor_queue.get()
                if item is None:  # shutdown sentinel
                    return
                yield item

        self._agent = Agent(
            name="live_agent",
            model=self._model,
            instruction=system_prompt,
            tools=[*_LIVE_AGENT_TOOLS, monitor],
            before_tool_callback=self._proxy_before_tool_callback,
        )

        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=self._agent,
            app_name=_APP_NAME,
            session_service=self._session_service,
        )

        await self._session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=self._session_id,
        )

        self._queue = LiveRequestQueue()
        self._event_loop_task = asyncio.create_task(self._run_event_loop())
        self._started = True

    async def stop(self) -> None:
        """Close queue and cancel event loop task."""
        if not self._started:
            return

        # Signal monitor() generator to exit
        try:
            self._monitor_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        if self._queue is not None:
            self._queue.close()
        if self._event_loop_task is not None:
            self._event_loop_task.cancel()
            try:
                await self._event_loop_task
            except asyncio.CancelledError:
                pass
            self._event_loop_task = None

        # Cancel any pending tool futures
        for call_id, future in self._tool_results.items():
            if not future.done():
                future.cancel()
        self._tool_results.clear()

        self._started = False
        self._state.connected = False

    def close_queue(self) -> None:
        """Close the live request queue (unblocks run_live)."""
        if self._queue is not None:
            self._queue.close()

    def on_audio_chunk(self, pcm_bytes: bytes) -> None:
        """Forward raw PCM audio to Gemini via send_realtime."""
        if self._queue is not None and self._state.connected:
            blob = types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000")
            self._queue.send_realtime(blob)

    def send_text(self, text: str) -> None:
        """Send a text message from the operator into the Gemini session."""
        if self._queue is not None and self._state.connected:
            content = types.Content(
                role="user",
                parts=[types.Part(text=text)],
            )
            self._queue.send_content(content)

    async def _proxy_before_tool_callback(self, tool, args, tool_context):
        """ADK before_tool_callback: proxy tool calls to the TUI via WebSocket.

        Returns a dict so ADK skips the actual tool function and uses
        the dict as the tool response. Returns None for ``monitor`` so ADK
        executes the real async generator.
        """
        if tool.name == "monitor":
            return None

        call_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._tool_results[call_id] = future

        # Fire callback — WS handler sends to TUI
        if self._on_tool_call:
            self._on_tool_call(call_id, tool.name, dict(args))

        # Await TUI response (with timeout)
        try:
            result = await asyncio.wait_for(future, timeout=_TOOL_TIMEOUT_S)
        except TimeoutError:
            logger.warning("Proxy tool %s timed out (call_id=%s)", tool.name, call_id)
            result = f"Tool {tool.name} timed out waiting for TUI response."
        finally:
            self._tool_results.pop(call_id, None)

        return {"result": result}

    def _build_run_config(self) -> RunConfig:
        """Build ADK RunConfig for the live agent session.

        Follows the pattern from google/adk-samples bidi-demo:
        native audio models use AUDIO response modality with transcription.
        """
        from google.adk.agents.run_config import StreamingMode

        return RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                    prefix_padding_ms=100,
                    silence_duration_ms=500,
                ),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),
        )

    async def _run_event_loop(self) -> None:
        """Background task: run_live() with reconnection on failure."""
        delay = 1.0
        while True:
            try:
                await self._run_live_session()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Live agent session crashed, reconnecting in %.1fs", delay)
                self._state.connected = False
                self._state.reconnect_count += 1
                await asyncio.sleep(delay)
                delay = min(delay * 2, _MAX_RECONNECT_DELAY_S)
                # Fresh session ID + queue so ADK doesn't replay stale history
                self._session_id = f"live_agent-{self._arm_id}-{uuid.uuid4().hex[:8]}"
                await self._session_service.create_session(
                    app_name=_APP_NAME,
                    user_id=_USER_ID,
                    session_id=self._session_id,
                )
                self._queue = LiveRequestQueue()
                continue
            break

    async def _run_live_session(self) -> None:
        """Single run_live() iteration — iterates events until the session ends."""
        if self._runner is None or self._queue is None:
            return

        run_config = self._build_run_config()
        self._state.connected = True

        # Kick the model to invoke monitor() immediately
        self._queue.send_content(
            types.Content(
                role="user",
                parts=[types.Part(text="Session started. Call monitor() now to receive robot updates.")],
            )
        )

        async for event in self._runner.run_live(
            user_id=_USER_ID,
            session_id=self._session_id,
            live_request_queue=self._queue,
            run_config=run_config,
        ):
            self._handle_event(event)

        self._state.connected = False

    def _handle_event(self, event) -> None:
        """Process a single ADK Event from the live stream."""
        # Log barge-in (server-side only — no callback to client)
        if event.interrupted:
            logger.info("Barge-in detected — interrupting audio output")

        # Handle content output (audio + text)
        if event.content and event.content.parts:
            for part in event.content.parts:
                # Audio data
                if part.inline_data and part.inline_data.mime_type and "audio" in part.inline_data.mime_type:
                    if self._on_audio_out and part.inline_data.data:
                        if not self._state.turn_active:
                            self._state.turn_active = True
                        self._on_audio_out(part.inline_data.data)
                # Text content (non-partial)
                elif part.text and not event.partial:
                    self._state.last_text_out = part.text
                    if self._on_text_out:
                        self._on_text_out(part.text)

        # Transcriptions — forward raw chunks to WS; client accumulates
        if event.input_transcription and event.input_transcription.text:
            finished = bool(getattr(event.input_transcription, "finished", False))
            self._state.last_transcription_in = event.input_transcription.text
            if self._on_transcription_in:
                self._on_transcription_in(event.input_transcription.text, finished)
        if event.output_transcription and event.output_transcription.text:
            finished = bool(getattr(event.output_transcription, "finished", False))
            self._state.last_transcription_out = event.output_transcription.text
            if self._on_transcription_out:
                self._on_transcription_out(event.output_transcription.text, finished)

        # Turn complete
        if event.turn_complete:
            self._state.turn_active = False
