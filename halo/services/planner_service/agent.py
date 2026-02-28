from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.planner_service.service import DecideFn
from halo.services.planner_service.snapshot_serializer import snapshot_to_dict
from halo.services.planner_service.tools import AgentContext, build_tools

_SNAPSHOT_PREFIX = "Current robot state:"
_DEPRECATED_CONTENT = "[DEPRECATED - superseded by a more recent snapshot]"
_THREAD_ID = "planner"  # single-thread; one conversation per PlannerAgent


@before_model
def _deprecate_old_snapshots(state: AgentState, runtime: object) -> dict | None:
    """Enforce the 'exactly one snapshot' invariant from the architecture spec.

    Before every model call, replace the content of all but the most recent
    "Current robot state:" HumanMessage with a deprecation notice so the LLM
    only ever reasons about the latest snapshot and the context stays small.
    """
    messages = state["messages"]
    indices = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.startswith(_SNAPSHOT_PREFIX)
    ]
    if len(indices) <= 1:
        return None
    # Replace all but last snapshot with deprecation notice
    updates = []
    for i in indices[:-1]:
        msg = messages[i]
        updates.append(RemoveMessage(id=msg.id))
        updates.append(HumanMessage(content=_DEPRECATED_CONTENT, id=msg.id))
    return {"messages": updates}


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


def _command_key(cmd: CommandEnvelope) -> str:
    """Stable key for a command (type + payload, ignoring IDs and timestamps)."""
    payload_dict = dataclasses.asdict(cmd.payload)
    return f"{cmd.type}:{json.dumps(payload_dict, sort_keys=True)}"


class PlannerAgent:
    """LangGraph ReAct agent that implements the DecideFn protocol."""

    MAX_LOOP_RETRIES = 4

    def __init__(
        self,
        model_name: str,
        base_url: str,
        prompts_dir: Path,
    ) -> None:
        system_prompt = _load_prompts(prompts_dir)
        self._ctx = AgentContext(arm_id="", snapshot_id=None)
        self._tools = build_tools(self._ctx)
        llm = ChatOllama(model=model_name, base_url=base_url)
        self._checkpointer = InMemorySaver()
        self._agent = create_agent(
            llm,
            self._tools,
            system_prompt=system_prompt,
            middleware=[_deprecate_old_snapshots],
            checkpointer=self._checkpointer,
        )
        self._last_reasoning: str = ""
        self._cmd_streak: dict[str, int] = {}  # command_key → consecutive count

    @property
    def last_reasoning(self) -> str:
        """Final LLM text from the most recent decide() call (empty if none)."""
        return self._last_reasoning

    def reset_loop_state(self) -> None:
        """Clear the loop-detection history (e.g. on new operator instruction)."""
        self._cmd_streak.clear()

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
    ) -> list[CommandEnvelope]:
        """DecideFn implementation. Thread-safe per design (never called concurrently).

        Uses InMemorySaver checkpointer to persist conversation history across
        calls.  The _deprecate_old_snapshots middleware replaces all but the
        latest snapshot with a short deprecation notice before each model call,
        keeping the context window manageable.

        Args:
            snap: Current runtime snapshot.
            operator_cmd: Optional natural-language instruction from the operator.
                          When provided it is appended as a second HumanMessage so
                          the agent can act on it in the context of the current state.
        """
        self._ctx.arm_id = snap.arm_id
        self._ctx.snapshot_id = snap.snapshot_id
        self._ctx.commands.clear()
        self._ctx.used_tools.clear()

        # New operator instruction always resets loop detection so the
        # agent gets a clean slate to act on the command.
        if operator_cmd:
            self._cmd_streak.clear()

        snap_json = json.dumps(snapshot_to_dict(snap), indent=2)

        messages: list = [
            HumanMessage(content=f"{_SNAPSHOT_PREFIX}\n```json\n{snap_json}\n```"),
        ]
        if operator_cmd:
            messages.append(HumanMessage(content=f"Operator: {operator_cmd}"))

        config = {"configurable": {"thread_id": _THREAD_ID}}
        result = await self._agent.ainvoke({"messages": messages}, config)
        self._last_reasoning = _extract_reasoning(result)

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
