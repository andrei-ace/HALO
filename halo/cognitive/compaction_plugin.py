"""CompactionPlugin — stable ADK plugin for snapshot deprecation + session compaction.

Replaces the experimental ``EventsCompactionConfig`` / ``App`` pattern with
stable ADK primitives:

- ``BasePlugin`` with ``before_model_callback`` and ``after_run_callback``
- ``EventCompaction``, ``EventActions``, ``Event`` — stable data types
- ``InvocationContext.session`` / ``InvocationContext.session_service``

ADK's ``_process_compaction_events()`` in the content pipeline automatically
replaces compacted events with the summary on the next ``run_async()`` call.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import litellm
from google.adk.events import Event, EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from halo.cognitive.compactor import CompactionResult, MessageHistory
from halo.cognitive.config import CompactionConfig

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse

logger = logging.getLogger(__name__)

_SNAPSHOT_PREFIX = "Current robot state:"
_DEPRECATED_CONTENT = "[DEPRECATED - superseded by a more recent snapshot]"

_SNAPSHOT_BLOCK_RE = re.compile(
    r"Current robot state:\s*```json\s*\n.*?```",
    re.DOTALL,
)

_SUMMARIZATION_PROMPT = (
    "The following is a conversation history between a user and an AI agent.\n"
    "Please summarize the conversation, focusing on key information and decisions\n"
    "made, as well as any unresolved questions or tasks. The summary should be\n"
    "concise and capture the essence of the interaction.\n\n"
    "{conversation_history}"
)


@dataclass
class _PendingHistoryCompaction:
    """Deferred MessageHistory compaction — applied after model response is tracked."""

    compacted_inv_count: int
    summary: str


class CompactionPlugin(BasePlugin):
    """ADK plugin: snapshot deprecation + optional session-level compaction.

    When ``config`` is None or ``config.enabled`` is False, only the
    ``before_model_callback`` (snapshot deprecation) runs.  When enabled,
    ``after_run_callback`` counts invocations and triggers LLM-based
    compaction, appending an ``EventCompaction`` event to the session.

    The MessageHistory compaction is **deferred** — ``after_run_callback``
    fires before the model response is appended to MessageHistory, so the
    actual ``apply_compaction`` must be called later via
    ``apply_deferred_history_compaction()`` once both user and model
    messages are tracked.

    Attributes:
        last_compaction: Set after deferred compaction is applied; read by
            ``PlannerAgent`` to propagate to the Switchboard.
    """

    def __init__(
        self,
        config: CompactionConfig | None = None,
        model_name: str = "",
        msg_history: MessageHistory | None = None,
    ) -> None:
        super().__init__(name="compaction")
        self._config = config
        self._model_name = model_name
        self._msg_history = msg_history
        self._last_compaction_end_ts: float = 0.0
        self.last_compaction: CompactionResult | None = None
        self._pending: _PendingHistoryCompaction | None = None

    @property
    def enabled(self) -> bool:
        return self._config is not None and self._config.enabled

    # ------------------------------------------------------------------
    # before_model_callback — snapshot deprecation
    # ------------------------------------------------------------------

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        """Enforce exactly-one-snapshot invariant.

        Strip the JSON snapshot block from all but the most recent
        ``Current robot state:`` user message.  Preserves operator
        instructions and other text.
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
            replaced = _SNAPSHOT_BLOCK_RE.sub(_DEPRECATED_CONTENT, old_text).strip()
            contents[i] = types.Content(
                role="user",
                parts=[types.Part.from_text(text=replaced or _DEPRECATED_CONTENT)],
            )
        return None

    # ------------------------------------------------------------------
    # after_run_callback — session-level compaction
    # ------------------------------------------------------------------

    async def after_run_callback(
        self,
        *,
        invocation_context: InvocationContext,
    ) -> None:
        """Count invocations since last compaction; trigger when threshold reached.

        NOTE: This callback fires *during* ``runner.run_async()``, before the
        caller has appended the model response to ``MessageHistory``.  We defer
        the ``MessageHistory.apply_compaction()`` to
        ``apply_deferred_history_compaction()`` which the caller must invoke
        after tracking the model response.
        """
        self.last_compaction = None
        self._pending = None

        if not self.enabled:
            return

        session = invocation_context.session
        events = session.events or []

        new_inv_ids = self._count_new_invocations(events, self._last_compaction_end_ts)
        if len(new_inv_ids) < self._config.compaction_interval:
            return

        try:
            to_compact, start_ts, end_ts = self._collect_events_to_compact(
                events, new_inv_ids, self._config.overlap_size
            )
            if not to_compact:
                return

            summary = await self._summarize_events(to_compact, self._model_name)
            if not summary:
                return

            # Use the root agent's name so ADK doesn't warn about unknown author.
            agent_name = invocation_context.agent.name if invocation_context.agent else "system"
            compaction_event = Event(
                author=agent_name,
                invocation_id=invocation_context.invocation_id,
                actions=EventActions(
                    compaction=EventCompaction(
                        start_timestamp=start_ts,
                        end_timestamp=end_ts,
                        compacted_content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=summary)],
                        ),
                    ),
                ),
            )
            await invocation_context.session_service.append_event(session, compaction_event)
            self._last_compaction_end_ts = end_ts

            # Defer MessageHistory compaction — model response not yet tracked.
            if self._msg_history is not None:
                compacted_inv_count = len(new_inv_ids) - self._config.overlap_size
                self._pending = _PendingHistoryCompaction(
                    compacted_inv_count=compacted_inv_count,
                    summary=summary,
                )

        except Exception:
            logger.debug("Compaction failed", exc_info=True)

    def apply_deferred_history_compaction(self) -> None:
        """Apply pending MessageHistory compaction after model response is tracked.

        Must be called by the agent after ``msg_history.append("model", ...)``
        so that the overlap window includes complete invocations (user + model).
        """
        if self._pending is None or self._msg_history is None:
            return

        try:
            boundary_id = self._find_compaction_boundary(self._msg_history, self._pending.compacted_inv_count)
            if boundary_id:
                self.last_compaction = self._msg_history.apply_compaction(boundary_id, self._pending.summary)
        except (ValueError, IndexError):
            logger.debug("apply_compaction failed", exc_info=True)
        finally:
            self._pending = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_new_invocations(events: list[Event], last_end_ts: float) -> list[str]:
        """Return unique invocation_ids from events after ``last_end_ts``."""
        seen: dict[str, None] = {}
        for ev in events:
            if ev.timestamp > last_end_ts and ev.invocation_id:
                if ev.invocation_id not in seen:
                    seen[ev.invocation_id] = None
        return list(seen)

    @staticmethod
    def _collect_events_to_compact(
        events: list[Event],
        invocation_ids: list[str],
        overlap_size: int,
    ) -> tuple[list[Event], float, float]:
        """Select events to compact (all except the last ``overlap_size`` invocations).

        Returns (events_to_compact, start_timestamp, end_timestamp).
        """
        if len(invocation_ids) <= overlap_size:
            return [], 0.0, 0.0

        compact_inv_ids = set(invocation_ids[:-overlap_size])
        to_compact = [ev for ev in events if ev.invocation_id in compact_inv_ids]
        if not to_compact:
            return [], 0.0, 0.0

        start_ts = min(ev.timestamp for ev in to_compact)
        end_ts = max(ev.timestamp for ev in to_compact)
        return to_compact, start_ts, end_ts

    @staticmethod
    def _find_compaction_boundary(msg_history: MessageHistory, compacted_inv_count: int) -> str | None:
        """Find the msg_id of the last message in the compacted range.

        Each invocation corresponds to one user message + one model response.
        We want to compact the first ``compacted_inv_count`` invocations'
        messages and retain the rest.  Returns the msg_id of the last record
        to compact (the model response of the last compacted invocation).
        """
        records = msg_history.get_all()
        # Count user messages as invocation markers
        user_count = 0
        last_in_range_idx = -1
        for i, rec in enumerate(records):
            if rec.role == "user":
                user_count += 1
                if user_count > compacted_inv_count:
                    break
            last_in_range_idx = i
        if last_in_range_idx < 0 or user_count < compacted_inv_count:
            return None
        return records[last_in_range_idx].msg_id

    @staticmethod
    async def _summarize_events(events: list[Event], model_name: str) -> str:
        """Call LLM via litellm to summarize compacted events."""
        text = CompactionPlugin._format_events_for_prompt(events)
        prompt = _SUMMARIZATION_PROMPT.format(conversation_history=text)

        # litellm needs "gemini/" prefix to route via GOOGLE_API_KEY;
        # bare "gemini-*" strings go to Vertex AI which requires different auth.
        litellm_model = model_name
        if model_name.startswith("gemini") and not model_name.startswith("gemini/"):
            litellm_model = f"gemini/{model_name}"

        response = await litellm.acompletion(
            model=litellm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _format_events_for_prompt(events: list[Event]) -> str:
        """Format events as text for the summarization prompt."""
        lines: list[str] = []
        for ev in events:
            role = ev.author or "unknown"
            text_parts: list[str] = []
            if ev.content and ev.content.parts:
                for part in ev.content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        text_parts.append(f"[tool_call: {fc.name}({fc.args})]")
                    elif hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        text_parts.append(f"[tool_result: {fr.name} -> {fr.response}]")
            if text_parts:
                lines.append(f"{role}: {' '.join(text_parts)}")
        return "\n".join(lines)
