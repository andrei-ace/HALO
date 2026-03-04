"""LivePlannerSession — persistent bidirectional streaming session via ADK run_live().

Bridges the request-response ``decide()`` pattern used by PlannerService
with the persistent WebSocket stream of the Gemini Live API.  Audio flows
continuously in parallel via ``send_realtime()``, while ``decide()`` pushes
snapshot content via ``send_content()`` and waits for ``turn_complete``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from google.adk.agents import Agent, LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from halo.cognitive.config import LiveConfig
from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.planner_service.snapshot_serializer import snapshot_to_dict
from halo.services.planner_service.tools import AgentContext, build_tools

logger = logging.getLogger(__name__)

_SNAPSHOT_PREFIX = "Current robot state:"
_APP_NAME = "halo-live"
_USER_ID = "planner"
_SESSION_ID = "live"
_DEFAULT_PROMPTS_DIR = Path(__file__).parents[2] / "configs" / "planner"
_DECIDE_TIMEOUT_S = 30.0
_MAX_RECONNECT_DELAY_S = 30.0


@dataclass
class LiveSessionState:
    """Observable state of the live session for TUI display."""

    connected: bool = False
    last_transcription_in: str = ""
    last_transcription_out: str = ""
    last_text_out: str = ""
    turn_active: bool = False
    reconnect_count: int = 0


def _load_prompts(prompts_dir: Path) -> str:
    """Load system_prompt.md + all skills/*.md and return combined string."""
    system_prompt_path = prompts_dir / "system_prompt.md"
    parts = [system_prompt_path.read_text(encoding="utf-8")]

    skills_dir = prompts_dir / "skills"
    if skills_dir.is_dir():
        skill_files = sorted(skills_dir.glob("*.md"))
        if skill_files:
            parts.append("\n\n# Skill Reference\n")
            for sf in skill_files:
                parts.append(sf.read_text(encoding="utf-8"))

    return "\n".join(parts)


def _command_key(cmd: CommandEnvelope) -> str:
    """Stable key for a command (type + payload, ignoring IDs and timestamps)."""
    payload_dict = dataclasses.asdict(cmd.payload)
    return f"{cmd.type}:{json.dumps(payload_dict, sort_keys=True)}"


class LivePlannerSession:
    """Persistent bidirectional streaming session to Gemini via ADK run_live().

    The session supports two parallel data flows:

    1. **Audio** (continuous): AudioCapture → send_realtime(Blob) → Gemini → audio
       response → AudioPlayback.  Audio flows independently of decide() calls.

    2. **Snapshot/commands** (request-response bridge): decide(snap) pushes snapshot
       text via send_content(), then awaits turn_complete.  Tool calls execute
       during the turn, accumulating commands in the AgentContext.
    """

    MAX_LOOP_RETRIES = 4

    def __init__(
        self,
        config: LiveConfig | None = None,
        prompts_dir: Path | None = None,
        audio_capture: object | None = None,
        audio_playback: object | None = None,
    ) -> None:
        self._config = config or LiveConfig()
        self._prompts_dir = prompts_dir or _DEFAULT_PROMPTS_DIR
        self._audio_capture = audio_capture
        self._audio_playback = audio_playback

        self._state = LiveSessionState()
        self._ctx = AgentContext(arm_id="", snapshot_id=None)
        self._tools = build_tools(self._ctx)

        # Built lazily in start()
        self._agent: Agent | None = None
        self._runner: Runner | None = None
        self._session_service: InMemorySessionService | None = None
        self._queue: LiveRequestQueue | None = None
        self._event_loop_task: asyncio.Task | None = None
        self._started = False

        # Turn synchronization
        self._turn_complete_event = asyncio.Event()
        self._pending_commands: asyncio.Queue[CommandEnvelope] = asyncio.Queue()

        # Loop detection
        self._last_reasoning: str = ""
        self._cmd_streak: dict[str, int] = {}

    @property
    def state(self) -> LiveSessionState:
        return self._state

    @property
    def last_reasoning(self) -> str:
        return self._last_reasoning

    def reset_loop_state(self) -> None:
        self._cmd_streak.clear()

    async def start(self) -> None:
        """Create ADK Agent, Runner, session, and start the event loop."""
        if self._started:
            return

        system_prompt = _load_prompts(self._prompts_dir)

        self._agent = Agent(
            name="planner-live",
            model=self._config.planner_model,
            instruction=system_prompt,
            tools=self._tools,
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
            session_id=_SESSION_ID,
        )

        self._queue = LiveRequestQueue()
        self._event_loop_task = asyncio.create_task(self._run_event_loop())
        self._started = True

        # Start audio capture if available
        if self._audio_capture is not None:
            try:
                self._audio_capture.start()
            except Exception:
                logger.exception("Failed to start audio capture")

        # Start audio playback if available
        if self._audio_playback is not None:
            try:
                self._audio_playback.start()
            except Exception:
                logger.exception("Failed to start audio playback")

    async def stop(self) -> None:
        """Stop audio, close queue, cancel event loop task."""
        if not self._started:
            return

        # Stop audio
        if self._audio_capture is not None:
            try:
                self._audio_capture.stop()
            except Exception:
                pass
        if self._audio_playback is not None:
            try:
                self._audio_playback.stop()
            except Exception:
                pass

        # Close queue and cancel event loop
        if self._queue is not None:
            self._queue.close()
        if self._event_loop_task is not None:
            self._event_loop_task.cancel()
            try:
                await self._event_loop_task
            except asyncio.CancelledError:
                pass
            self._event_loop_task = None

        self._started = False
        self._state.connected = False

    def on_audio_chunk(self, pcm_bytes: bytes) -> None:
        """Callback from AudioCapture — forward raw PCM to Gemini via send_realtime."""
        if self._queue is not None and self._state.connected:
            blob = types.Blob(data=pcm_bytes, mime_type=f"audio/pcm;rate={self._config.input_sample_rate}")
            self._queue.send_realtime(blob)

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        """Push snapshot content, wait for turn_complete, return accumulated commands.

        This bridges the request-response pattern expected by PlannerService
        with the persistent streaming session.
        """
        if not self._started:
            await self.start()

        # 1. Drain pre-existing commands (voice-triggered between ticks)
        pre_commands = self._drain_pending_queue()

        # 2. Update context
        self._ctx.arm_id = snap.arm_id
        self._ctx.snapshot_id = snap.snapshot_id
        self._ctx.epoch = epoch
        self._ctx.commands.clear()
        self._ctx.used_tools.clear()

        # Reset loop detection on new operator instruction
        if operator_cmd:
            self._cmd_streak.clear()

        # 3. Build content
        snap_json = json.dumps(snapshot_to_dict(snap), indent=2)
        text_parts = [f"{_SNAPSHOT_PREFIX}\n```json\n{snap_json}\n```"]
        if operator_cmd:
            text_parts.append(f"Operator: {operator_cmd}")

        content = types.Content(
            role="user",
            parts=[types.Part(text="\n\n".join(text_parts))],
        )

        # 4. Send content and wait for turn_complete
        self._turn_complete_event.clear()
        self._state.turn_active = True

        if self._queue is not None:
            self._queue.send_content(content)

        try:
            await asyncio.wait_for(self._turn_complete_event.wait(), timeout=_DECIDE_TIMEOUT_S)
        except TimeoutError:
            logger.warning("decide() timed out waiting for turn_complete")
            self._state.turn_active = False

        self._state.turn_active = False

        # 5. Drain post-turn commands
        turn_commands = list(self._ctx.commands)
        post_commands = self._drain_pending_queue()

        commands = pre_commands + turn_commands + post_commands

        # 6. Loop detection
        if commands:
            current_keys = {_command_key(c) for c in commands}
            for key in current_keys:
                self._cmd_streak[key] = self._cmd_streak.get(key, 0) + 1
            for key in list(self._cmd_streak):
                if key not in current_keys:
                    del self._cmd_streak[key]
            max_streak = max(self._cmd_streak.values())
            if max_streak >= self.MAX_LOOP_RETRIES:
                looped = [k.split(":")[0] for k, v in self._cmd_streak.items() if v >= self.MAX_LOOP_RETRIES]
                self._last_reasoning = (
                    f"Loop detected: {', '.join(looped)} issued "
                    f"{self.MAX_LOOP_RETRIES}+ times in a row. "
                    f"Stopping to avoid an infinite loop. "
                    f"Awaiting operator intervention."
                )
                self._cmd_streak.clear()
                return []

        return commands

    def drain_pending_commands(self) -> list[CommandEnvelope]:
        """Public API: drain voice-triggered commands accumulated between ticks."""
        return self._drain_pending_queue()

    def _drain_pending_queue(self) -> list[CommandEnvelope]:
        commands = []
        while not self._pending_commands.empty():
            try:
                commands.append(self._pending_commands.get_nowait())
            except asyncio.QueueEmpty:
                break
        return commands

    def _build_run_config(self) -> RunConfig:
        """Build ADK RunConfig for the live session."""
        cfg = self._config

        speech_config = None
        if cfg.audio_enabled:
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=cfg.voice_name),
                ),
            )

        modalities = list(cfg.response_modalities) if cfg.audio_enabled else ["TEXT"]

        session_resumption = None
        if cfg.session_resumption:
            session_resumption = types.SessionResumptionConfig(transparent=True)

        context_compression = None
        if cfg.context_compression:
            context_compression = types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            )

        transcription_config = types.AudioTranscriptionConfig() if cfg.enable_transcription else None

        return RunConfig(
            response_modalities=modalities,
            speech_config=speech_config,
            session_resumption=session_resumption,
            context_window_compression=context_compression,
            output_audio_transcription=transcription_config,
            input_audio_transcription=transcription_config,
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
                logger.exception("Live session crashed, reconnecting in %.1fs", delay)
                self._state.connected = False
                self._state.reconnect_count += 1
                await asyncio.sleep(delay)
                delay = min(delay * 2, _MAX_RECONNECT_DELAY_S)
                # Re-create queue for new session
                self._queue = LiveRequestQueue()
                continue
            break  # Clean exit (e.g. queue closed)

    async def _run_live_session(self) -> None:
        """Single run_live() iteration — iterates events until the session ends."""
        if self._runner is None or self._queue is None:
            return

        run_config = self._build_run_config()
        self._state.connected = True

        async for event in self._runner.run_live(
            user_id=_USER_ID,
            session_id=_SESSION_ID,
            live_request_queue=self._queue,
            run_config=run_config,
        ):
            self._handle_event(event)

        self._state.connected = False

    def _handle_event(self, event) -> None:
        """Process a single ADK Event from the live stream."""
        # Handle interruption — clear playback buffer
        if event.interrupted:
            if self._audio_playback is not None:
                self._audio_playback.clear()

        # Handle audio output
        if event.content and event.content.parts:
            for part in event.content.parts:
                # Audio data
                if part.inline_data and part.inline_data.mime_type and "audio" in part.inline_data.mime_type:
                    if self._audio_playback is not None and part.inline_data.data:
                        self._audio_playback.enqueue(part.inline_data.data)
                # Text content (non-partial)
                elif part.text and not event.partial:
                    self._last_reasoning = part.text
                    self._state.last_text_out = part.text

        # Transcriptions
        if event.input_transcription and event.input_transcription.text:
            self._state.last_transcription_in = event.input_transcription.text
        if event.output_transcription and event.output_transcription.text:
            self._state.last_transcription_out = event.output_transcription.text

        # Turn complete — drain tool-call commands and signal decide()
        if event.turn_complete:
            # Move commands from ctx to pending queue
            for cmd in self._ctx.commands:
                self._pending_commands.put_nowait(cmd)
            self._ctx.commands.clear()
            self._ctx.used_tools.clear()
            self._turn_complete_event.set()
