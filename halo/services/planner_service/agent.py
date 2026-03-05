from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from pathlib import Path

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from halo.cognitive.compactor import CompactionResult, MessageHistory
from halo.contracts.commands import CommandEnvelope
from halo.contracts.events import EventType
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.planner_service.service import DecideFn
from halo.services.planner_service.snapshot_serializer import snapshot_to_dict
from halo.services.planner_service.tools import AgentContext, build_tools

# Suppress the "non-text parts in the response" warning from google.genai.
# ADK's internal _build_response_log accesses resp.text on responses that
# contain function_call parts, triggering this harmless warning every call.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

_SNAPSHOT_PREFIX = "Current robot state:"
_DEPRECATED_CONTENT = "[DEPRECATED - superseded by a more recent snapshot]"
_THREAD_ID = "planner"  # single-thread; one conversation per PlannerAgent
_APP_NAME = "halo"
_USER_ID = "planner"


_SNAPSHOT_BLOCK_RE = re.compile(
    r"Current robot state:\s*```json\s*\n.*?```",
    re.DOTALL,
)


def _deprecate_old_snapshots(
    callback_context,  # noqa: ARG001 — required by ADK callback signature
    llm_request,
):
    """ADK before_model_callback: enforce exactly-one-snapshot invariant.

    Before every model call, strip the JSON snapshot block from all but the
    most recent "Current robot state:" user message.  Everything else in the
    message (operator instructions, event context) is preserved so the agent
    keeps task context across ticks.

    Returns None to let the (modified) request proceed to the model.
    """
    contents = llm_request.contents or []
    indices = [
        i
        for i, c in enumerate(contents)
        if c.role == "user"
        and c.parts
        and any(getattr(p, "text", None) and p.text.startswith(_SNAPSHOT_PREFIX) for p in c.parts)
    ]
    if len(indices) <= 1:
        return None
    for i in indices[:-1]:
        old_text = ""
        for p in contents[i].parts:
            t = getattr(p, "text", None)
            if t:
                old_text = t
                break
        # Strip the snapshot JSON block, keep everything else
        replaced = _SNAPSHOT_BLOCK_RE.sub(_DEPRECATED_CONTENT, old_text).strip()
        contents[i] = types.Content(
            role="user",
            parts=[types.Part.from_text(text=replaced or _DEPRECATED_CONTENT)],
        )
    return None


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


def _extract_scene_image(snap: PlannerSnapshot) -> object | None:
    """Extract the VLM image from the most recent SCENE_DESCRIBED event, if any."""
    for ev in reversed(snap.recent_events):
        if ev.type == EventType.SCENE_DESCRIBED and ev.data.get("vlm_image") is not None:
            return ev.data["vlm_image"]
    return None


def _to_png_bytes(image: object) -> bytes | None:
    """Convert numpy BGR, PIL, or raw bytes to PNG bytes for Gemini multimodal input."""
    import io

    import numpy as np
    from PIL import Image

    pil: Image.Image | None = None
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, np.ndarray):
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image))

    if pil is None:
        return None
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _command_key(cmd: CommandEnvelope) -> str:
    """Stable key for a command (type + payload, ignoring IDs and timestamps)."""
    payload_dict = dataclasses.asdict(cmd.payload)
    return f"{cmd.type}:{json.dumps(payload_dict, sort_keys=True)}"


class PlannerAgent:
    """ADK ReAct agent that implements the DecideFn protocol."""

    MAX_LOOP_RETRIES = 4

    def __init__(
        self,
        model_name: str,
        base_url: str,
        prompts_dir: Path,
        backend: str = "local",
        compaction_config: object | None = None,
    ) -> None:
        system_prompt = _load_prompts(prompts_dir)
        self._ctx = AgentContext(arm_id="", snapshot_id=None)
        self._tools = build_tools(self._ctx)
        self._backend = backend

        if backend == "cloud":
            # ADK natively routes bare model strings (e.g. "gemini-2.5-flash")
            # to the Gemini API via GOOGLE_API_KEY env var.
            model = model_name
        else:
            # LiteLLM uses OLLAMA_API_BASE env var for the Ollama endpoint.
            os.environ["OLLAMA_API_BASE"] = base_url
            model = LiteLlm(model=f"ollama_chat/{model_name}")

        self._agent = Agent(
            name="planner",
            model=model,
            instruction=system_prompt,
            tools=self._tools,
            before_model_callback=_deprecate_old_snapshots,
        )
        self._session_service = InMemorySessionService()

        # Cloud backend: use App pattern for ADK event compaction
        if backend == "cloud" and compaction_config is not None:
            from halo.cognitive.config import CompactionConfig

            cfg = compaction_config if isinstance(compaction_config, CompactionConfig) else CompactionConfig()
            if cfg.enabled:
                from google.adk.apps import App
                from google.adk.apps.app import EventsCompactionConfig

                app = App(
                    name=_APP_NAME,
                    root_agent=self._agent,
                    events_compaction_config=EventsCompactionConfig(
                        compaction_interval=cfg.compaction_interval,
                        overlap_size=cfg.overlap_size,
                    ),
                )
                self._runner = Runner(
                    app=app,
                    session_service=self._session_service,
                )
            else:
                self._runner = Runner(
                    agent=self._agent,
                    app_name=_APP_NAME,
                    session_service=self._session_service,
                )
        else:
            self._runner = Runner(
                agent=self._agent,
                app_name=_APP_NAME,
                session_service=self._session_service,
            )

        self._session_created = False
        self._last_reasoning: str = ""
        self._cmd_streak: dict[str, int] = {}  # command_key → consecutive count
        self._pending_handoff: str | None = None
        self._msg_history = MessageHistory()
        self._last_compaction: CompactionResult | None = None

    async def _ensure_session(self) -> None:
        """Create the ADK session on first use (requires async)."""
        if not self._session_created:
            await self._session_service.create_session(
                app_name=_APP_NAME,
                user_id=_USER_ID,
                session_id=_THREAD_ID,
            )
            self._session_created = True

    @property
    def last_reasoning(self) -> str:
        """Final LLM text from the most recent decide() call (empty if none)."""
        return self._last_reasoning

    @property
    def msg_history(self) -> MessageHistory:
        return self._msg_history

    @property
    def last_compaction(self) -> CompactionResult | None:
        """Non-None when the most recent decide() triggered ADK compaction."""
        return self._last_compaction

    def reset_loop_state(self) -> None:
        """Clear the loop-detection history (e.g. on new operator instruction)."""
        self._cmd_streak.clear()

    async def inject_handoff_context(self, context_text: str) -> None:
        """Queue handoff context to be prepended to the next decide() call."""
        self._pending_handoff = context_text

    async def reset_session(self) -> None:
        """Delete the current ADK session for a fresh start after backend switch."""
        if self._session_created:
            try:
                await self._session_service.delete_session(
                    app_name=_APP_NAME,
                    user_id=_USER_ID,
                    session_id=_THREAD_ID,
                )
            except Exception:
                pass
            self._session_created = False
        self._cmd_streak.clear()
        self._msg_history.clear()
        self._last_compaction = None

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        """DecideFn implementation. Thread-safe per design (never called concurrently).

        Uses InMemorySessionService to persist conversation history across
        calls.  The _deprecate_old_snapshots before_model_callback replaces all
        but the latest snapshot with a short deprecation notice before each
        model call, keeping the context window manageable.

        Args:
            snap: Current runtime snapshot.
            operator_cmd: Optional natural-language instruction from the operator.
                          When provided it is appended to the snapshot message so
                          the agent can act on it in the context of the current state.
                          Also resets loop detection.
            epoch: Optional lease epoch to stamp on generated commands.
        """
        await self._ensure_session()

        self._ctx.arm_id = snap.arm_id
        self._ctx.snapshot_id = snap.snapshot_id
        self._ctx.epoch = epoch
        self._ctx.commands.clear()
        self._ctx.used_tools.clear()

        # New operator instruction always resets loop detection so the
        # agent gets a clean slate to act on the command.
        if operator_cmd:
            self._cmd_streak.clear()

        snap_json = json.dumps(snapshot_to_dict(snap), indent=2)

        # Combine snapshot + optional operator command into one user message.
        # ADK's runner.run_async takes a single new_message per call.
        text_parts = [f"{_SNAPSHOT_PREFIX}\n```json\n{snap_json}\n```"]
        if self._pending_handoff:
            text_parts.insert(0, self._pending_handoff)
            self._pending_handoff = None
        if operator_cmd:
            text_parts.append(f"Operator: {operator_cmd}")

        parts: list = [types.Part.from_text(text="\n\n".join(text_parts))]

        # Cloud backend: attach the VLM scene image so Gemini can see what
        # the VLM analysed.  Local (Ollama) skips this — it doesn't support
        # inline images in the chat context.
        if self._backend == "cloud":
            scene_image = _extract_scene_image(snap)
            if scene_image is not None:
                png_bytes = _to_png_bytes(scene_image)
                if png_bytes is not None:
                    parts.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))

        message = types.Content(role="user", parts=parts)

        # Track user message in parallel history
        user_text = "\n\n".join(text_parts)
        user_msg_id = self._msg_history.append("user", user_text)

        # Snapshot session event count before run (for compaction detection)
        pre_event_count = 0
        if self._session_created:
            try:
                session = await self._session_service.get_session(
                    app_name=_APP_NAME,
                    user_id=_USER_ID,
                    session_id=_THREAD_ID,
                )
                if session:
                    pre_event_count = len(session.events)
            except Exception:
                pass

        # Run the agent and extract the final response text.
        last_text = ""
        async for event in self._runner.run_async(
            user_id=_USER_ID,
            session_id=_THREAD_ID,
            new_message=message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        last_text = part.text

        self._last_reasoning = last_text

        # Track model response
        if last_text:
            self._msg_history.append("model", last_text)

        # Detect ADK compaction: check for EventCompaction in new session events
        self._last_compaction = None
        if self._backend == "cloud" and self._session_created:
            self._last_compaction = await self._detect_compaction(pre_event_count, user_msg_id)

        commands = list(self._ctx.commands)

        # Loop detection: track how many consecutive times each command
        # (type+payload) has been issued. If any command hits the limit,
        # suppress the entire batch. This catches loops even when the
        # agent varies the number of commands per tick.
        if commands:
            current_keys = {_command_key(c) for c in commands}
            # Increment streaks for commands seen this tick
            for key in current_keys:
                self._cmd_streak[key] = self._cmd_streak.get(key, 0) + 1
            # Decay streaks for commands NOT seen this tick
            for key in list(self._cmd_streak):
                if key not in current_keys:
                    del self._cmd_streak[key]
            # Check if any command hit the limit
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

    async def _detect_compaction(self, pre_event_count: int, last_user_msg_id: str) -> CompactionResult | None:
        """Check session events for a new EventCompaction marker.

        ADK appends a compaction event at the end of ``run_async()`` when
        ``EventsCompactionConfig`` triggers.  We detect it by comparing
        event counts before and after, then looking for the ``compaction``
        field on new events.
        """
        try:
            session = await self._session_service.get_session(
                app_name=_APP_NAME,
                user_id=_USER_ID,
                session_id=_THREAD_ID,
            )
            if session is None:
                return None

            new_events = session.events[pre_event_count:]
            for ev in new_events:
                compaction = ev.actions.compaction if ev.actions else None
                if compaction is None:
                    continue
                # Extract summary from compacted_content
                summary_text = ""
                if compaction.compacted_content and compaction.compacted_content.parts:
                    for part in compaction.compacted_content.parts:
                        if hasattr(part, "text") and part.text:
                            summary_text += part.text
                if summary_text:
                    return self._msg_history.apply_compaction(last_user_msg_id, summary_text)
        except Exception:
            logging.getLogger(__name__).debug("Compaction detection failed", exc_info=True)
        return None


def make_decide_fn(
    model_name: str = "gpt-oss:20b",
    base_url: str = "http://localhost:11434",
    prompts_dir: Path | str | None = None,
    backend: str = "local",
) -> DecideFn:
    """Factory that creates a PlannerAgent and returns its decide method.

    Usage::

        from halo.services.planner_service.agent import make_decide_fn
        decide = make_decide_fn()                        # local Ollama
        decide = make_decide_fn(backend="cloud",
                                model_name="gemini-2.5-flash")  # Gemini
        svc = PlannerService("arm0", runtime, decide)
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parents[3] / "configs" / "planner"
    agent = PlannerAgent(model_name, base_url, Path(prompts_dir), backend=backend)
    return agent.decide
