from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.planner_service.snapshot_serializer import snapshot_to_dict
from halo.services.planner_service.tools import AgentContext, build_tools

DecideFn = Callable[[PlannerSnapshot], Awaitable[list[CommandEnvelope]]]


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
        tools = build_tools(self._ctx)
        llm = ChatOllama(model=model_name, base_url=base_url)
        self._agent = create_react_agent(llm, tools)

    async def decide(self, snap: PlannerSnapshot) -> list[CommandEnvelope]:
        """DecideFn implementation. Thread-safe per design (never called concurrently)."""
        self._ctx.arm_id = snap.arm_id
        self._ctx.snapshot_id = snap.snapshot_id
        self._ctx.commands.clear()

        snap_json = json.dumps(snapshot_to_dict(snap), indent=2)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=f"Current robot state:\n```json\n{snap_json}\n```"),
        ]
        await self._agent.ainvoke({"messages": messages})
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
