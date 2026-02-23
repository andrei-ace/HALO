from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.planner_service.snapshot_serializer import snapshot_to_dict
from halo.services.planner_service.tools import AgentContext, build_tools

DecideFn = Callable[[PlannerSnapshot], Awaitable[list[CommandEnvelope]]]

_SNAPSHOT_PREFIX = "Current robot state:"
_DEPRECATED_CONTENT = "[DEPRECATED - superseded by a more recent snapshot]"


class _DeprecateOldSnapshotsMiddleware(AgentMiddleware):
    """Enforce the 'exactly one snapshot' invariant from the architecture spec.

    Before every model call, replace the content of all but the most recent
    "Current robot state:" HumanMessage with a deprecation notice so the LLM
    only ever reasons about the latest snapshot and the context stays small.
    """

    @staticmethod
    def _deprecate(messages: list) -> list:
        indices = [
            i for i, m in enumerate(messages)
            if isinstance(m, HumanMessage)
            and isinstance(m.content, str)
            and m.content.startswith(_SNAPSHOT_PREFIX)
        ]
        for i in indices[:-1]:
            messages[i] = HumanMessage(content=_DEPRECATED_CONTENT)
        return messages

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        return handler(request.override(messages=self._deprecate(list(request.messages))))

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        return await handler(request.override(messages=self._deprecate(list(request.messages))))


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


def _extract_reasoning(result: object) -> str:
    """Pull the final AI text from an ainvoke result dict.

    The result is ``{"messages": [...]}``; we walk backwards to find the last
    AIMessage that has non-empty text content and no pending tool_calls.
    """
    try:
        msgs = result.get("messages", []) if isinstance(result, dict) else []  # type: ignore[union-attr]
        for msg in reversed(msgs):
            content = getattr(msg, "content", "")
            if content and isinstance(content, str) and not getattr(msg, "tool_calls", None):
                return content
    except Exception:
        pass
    return ""


class PlannerAgent:
    """LangGraph ReAct agent that implements the DecideFn protocol."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        prompts_dir: Path,
    ) -> None:
        self._system_prompt = _load_prompts(prompts_dir)
        self._ctx = AgentContext(arm_id="", snapshot_id=None)
        self._tools = build_tools(self._ctx)
        llm = ChatOllama(model=model_name, base_url=base_url)
        self._agent = create_agent(
            llm,
            self._tools,
            middleware=[_DeprecateOldSnapshotsMiddleware()],
        )
        self._last_reasoning: str = ""

    @property
    def last_reasoning(self) -> str:
        """Final LLM text from the most recent decide() call (empty if none)."""
        return self._last_reasoning

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
    ) -> list[CommandEnvelope]:
        """DecideFn implementation. Thread-safe per design (never called concurrently).

        Args:
            snap: Current runtime snapshot.
            operator_cmd: Optional natural-language instruction from the operator.
                          When provided it is appended as a second HumanMessage so
                          the agent can act on it in the context of the current state.
        """
        self._ctx.arm_id = snap.arm_id
        self._ctx.snapshot_id = snap.snapshot_id
        self._ctx.commands.clear()

        snap_json = json.dumps(snapshot_to_dict(snap), indent=2)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=f"{_SNAPSHOT_PREFIX}\n```json\n{snap_json}\n```"),
        ]
        if operator_cmd:
            messages.append(HumanMessage(content=f"Operator: {operator_cmd}"))
        result = await self._agent.ainvoke({"messages": messages})
        self._last_reasoning = _extract_reasoning(result)
        return list(self._ctx.commands)


def make_decide_fn(
    model_name: str = "gpt-oss:20B",
    base_url: str = "http://localhost:11434",
    prompts_dir: Path | str | None = None,
) -> DecideFn:
    """Factory that creates a PlannerAgent and returns its decide method.

    Usage::

        from halo.services.planner_service.agent import make_decide_fn
        decide = make_decide_fn()          # defaults to local Ollama
        svc = PlannerService("arm0", runtime, decide)
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parents[3] / "configs" / "planner"
    agent = PlannerAgent(model_name, base_url, Path(prompts_dir))
    return agent.decide
