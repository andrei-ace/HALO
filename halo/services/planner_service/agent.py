from __future__ import annotations

import contextvars
import dataclasses
import json
import logging
import os
import uuid
import weakref
from pathlib import Path

import litellm
from google.adk.agents import Agent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.apps import App
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from halo.cognitive.compaction_plugin import CompactionPlugin
from halo.cognitive.compactor import CompactionResult, MessageHistory, MessageRecord
from halo.cognitive.config import CompactionConfig
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
_THREAD_ID = "planner"  # single-thread; one conversation per PlannerAgent
_APP_NAME = "halo"
_USER_ID = "planner"

# Per-decide() invocation nonce.  The litellm success_callback is global, so we
# use a ContextVar to scope captured usage to the async task that owns a given
# decide() call.  Concurrent decide() calls (multi-arm) and unrelated litellm
# requests (compaction) run in different contexts and will not match.
_decide_nonce: contextvars.ContextVar[str] = contextvars.ContextVar("_decide_nonce", default="")


def _load_prompts(prompts_dir: Path) -> str:
    """Load system_prompt.md + skill prompts from configs/skills/*/system_prompt.md."""
    system_prompt_path = prompts_dir / "system_prompt.md"
    parts = [system_prompt_path.read_text(encoding="utf-8")]

    # Skill prompts live under configs/skills/<skill_name>/system_prompt.md
    skills_dir = prompts_dir.parent / "skills"
    if skills_dir.is_dir():
        skill_files = sorted(skills_dir.glob("*/system_prompt.md"))
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
        self._model_name = model_name

        if backend == "cloud":
            # ADK natively routes bare model strings (e.g. "gemini-3.1-flash-lite-preview")
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
        )
        self._session_service = InMemorySessionService()
        self._msg_history = MessageHistory()
        self._last_compaction: CompactionResult | None = None

        self._run_config = RunConfig(max_llm_calls=25)

        # Build CompactionPlugin — handles snapshot deprecation (all backends)
        # and optional session compaction (cloud backend with config).
        cfg: CompactionConfig | None = None
        if backend == "cloud" and compaction_config is not None:
            cfg = compaction_config if isinstance(compaction_config, CompactionConfig) else CompactionConfig()
        self._compaction_plugin = CompactionPlugin(
            config=cfg,
            model_name=model_name,
            msg_history=self._msg_history,
            agent_ctx=self._ctx,
        )

        app = App(
            name=_APP_NAME,
            root_agent=self._agent,
            plugins=[self._compaction_plugin],
        )
        self._runner = Runner(
            app=app,
            session_service=self._session_service,
        )

        self._session_created = False
        self._last_reasoning: str = ""
        self._last_token_usage: dict[str, int] = {}
        self._cmd_streak: dict[str, int] = {}  # command_key → consecutive count

        # litellm fallback: a permanent callback scoped by ContextVar nonce.
        # Captures token usage when ADK's event.usage_metadata is empty
        # (e.g. Ollama via LiteLlm).  Uses a weakref so the callback does not
        # prevent GC if the agent is discarded (tests, failover re-init).
        self._litellm_usage: dict[str, int] = {}
        self._litellm_nonce: str = ""

        ref = weakref.ref(self)

        def _on_litellm_success(kwargs, response_obj, start_time, end_time):  # noqa: ARG001
            agent = ref()
            if agent is None or _decide_nonce.get("") != agent._litellm_nonce:
                return
            usage = getattr(response_obj, "usage", None)
            if usage:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    agent._litellm_usage[key] = agent._litellm_usage.get(key, 0) + getattr(usage, key, 0)

        self._litellm_callback = _on_litellm_success  # prevent callback GC while agent lives
        litellm.success_callback.append(_on_litellm_success)
        self._pending_handoff: str | None = None

    def __del__(self) -> None:
        """Remove our callback from the global litellm list on GC."""
        try:
            litellm.success_callback.remove(self._litellm_callback)
        except (ValueError, AttributeError):
            pass

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
    def model_name(self) -> str:
        return self._model_name

    @property
    def last_reasoning(self) -> str:
        """Final LLM text from the most recent decide() call (empty if none)."""
        return self._last_reasoning

    @property
    def last_token_usage(self) -> dict[str, int]:
        """Token usage from the most recent decide() call (empty if unavailable)."""
        return self._last_token_usage

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

    async def inject_compaction_state(self, summary: str, retained_records: list[MessageRecord]) -> None:
        """Rebuild session and MessageHistory from a compaction summary + retained records.

        Used by ``Switchboard._sync_compaction_to_inactive`` to propagate cloud
        compaction state to the local backend so that on failover the local model
        starts with proper conversation history instead of just a text prefix.

        Steps:
        1. Reset session (wipe old ADK state + MessageHistory)
        2. Create fresh session
        3. Inject summary as a model Event
        4. Inject each retained user+model pair as Events
        5. Mirror all records into MessageHistory
        """
        await self.reset_session()
        self._pending_handoff = None
        await self._ensure_session()

        session = await self._session_service.get_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=_THREAD_ID,
        )

        # Inject summary as model event when one exists. Empty summary means
        # "replay the raw retained transcript" rather than inventing a blank turn.
        if summary:
            summary_inv_id = uuid.uuid4().hex
            summary_event = Event(
                author="planner",
                invocation_id=summary_inv_id,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=summary)],
                ),
            )
            await self._session_service.append_event(session, summary_event)
            self._msg_history.append("model", summary)
            # Mark summary record
            recs = self._msg_history.get_all()
            if recs:
                last = recs[-1]
                # Replace with is_summary=True variant
                self._msg_history._records[-1] = MessageRecord(
                    msg_id=last.msg_id,
                    role=last.role,
                    text=last.text,
                    ts_ms=last.ts_ms,
                    is_summary=True,
                )

        # Inject retained records as user+model pairs
        i = 0
        while i < len(retained_records):
            rec = retained_records[i]
            inv_id = uuid.uuid4().hex

            if rec.role == "user":
                # Inject user event
                user_event = Event(
                    author="user",
                    invocation_id=inv_id,
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=rec.text)],
                    ),
                )
                await self._session_service.append_event(session, user_event)
                self._msg_history.append("user", rec.text)

                # Check for paired model response
                if i + 1 < len(retained_records) and retained_records[i + 1].role == "model":
                    model_rec = retained_records[i + 1]
                    model_event = Event(
                        author="planner",
                        invocation_id=inv_id,
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=model_rec.text)],
                        ),
                    )
                    await self._session_service.append_event(session, model_event)
                    self._msg_history.append("model", model_rec.text)
                    i += 2
                else:
                    i += 1
            elif rec.role == "model":
                # Unpaired model record — inject standalone
                model_event = Event(
                    author="planner",
                    invocation_id=inv_id,
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=rec.text)],
                    ),
                )
                await self._session_service.append_event(session, model_event)
                self._msg_history.append("model", rec.text)
                i += 1
            else:
                i += 1

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
        self._ctx.call_counts.clear()
        self._ctx.loop_detected = False

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
        self._msg_history.append("user", user_text)

        # Run the agent and extract the final response text + token usage.
        last_text = ""
        token_usage: dict[str, int] = {}

        # Activate litellm usage capture for this invocation via ContextVar nonce.
        self._litellm_usage.clear()
        self._litellm_nonce = uuid.uuid4().hex
        _decide_nonce.set(self._litellm_nonce)

        try:
            async for event in self._runner.run_async(
                user_id=_USER_ID,
                session_id=_THREAD_ID,
                new_message=message,
                run_config=self._run_config,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            last_text = part.text
                # Accumulate token usage across events (last event with data wins for
                # cumulative fields; ADK may emit partial usage on intermediate events).
                um = event.usage_metadata
                if um is not None:
                    if um.prompt_token_count is not None:
                        token_usage["prompt_tokens"] = um.prompt_token_count
                    if um.candidates_token_count is not None:
                        token_usage["completion_tokens"] = um.candidates_token_count
                    if um.total_token_count is not None:
                        token_usage["total_tokens"] = um.total_token_count
                    if um.thoughts_token_count:
                        token_usage["thoughts_tokens"] = um.thoughts_token_count
                    if um.cached_content_token_count:
                        token_usage["cached_tokens"] = um.cached_content_token_count
        except LlmCallsLimitExceededError:
            logging.getLogger(__name__).warning("LLM call limit exceeded — stopping agent loop")
            last_text = (
                "LLM call limit exceeded — agent was looping. "
                "Stopping to avoid runaway costs. Awaiting operator intervention."
            )
            self._ctx.commands.clear()

        # Deactivate capture; fall back to litellm data if ADK provided nothing.
        self._litellm_nonce = ""
        if not token_usage and self._litellm_usage:
            token_usage = dict(self._litellm_usage)

        self._last_reasoning = last_text
        self._last_token_usage = token_usage

        # Track model response
        if last_text:
            self._msg_history.append("model", last_text)

        # Apply deferred MessageHistory compaction now that both user and
        # model messages are tracked — ensures the overlap window contains
        # complete invocations (user + model pairs).
        self._compaction_plugin.apply_deferred_history_compaction()
        self._last_compaction = self._compaction_plugin.last_compaction

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
                                model_name="gemini-3.1-flash-lite-preview")  # Gemini
        svc = PlannerService("arm0", runtime, decide)
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parents[3] / "configs" / "planner"
    agent = PlannerAgent(model_name, base_url, Path(prompts_dir), backend=backend)
    return agent.decide
