"""Local (Ollama) cognitive backend — wraps PlannerAgent + Ollama VLM."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from halo.cognitive.config import BackendReadiness, BackendType, LocalConfig
from halo.cognitive.context_store import CognitiveState, ContextEntry
from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.planner_service.agent import PlannerAgent
from halo.services.target_perception_service.vlm_fn import make_vlm_fn
from halo.services.target_perception_service.vlm_parser import VlmScene

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

_DEFAULT_PROMPTS_DIR = Path(__file__).parents[2] / "configs" / "planner"


class LocalCognitiveBackend:
    """Brain + eyes backed by local Ollama."""

    def __init__(
        self,
        config: LocalConfig | None = None,
        prompts_dir: Path | None = None,
        run_logger: RunLogger | None = None,
    ) -> None:
        cfg = config or LocalConfig()
        self._config = cfg
        self._agent = PlannerAgent(
            model_name=cfg.planner_model,
            base_url=cfg.base_url,
            prompts_dir=prompts_dir or _DEFAULT_PROMPTS_DIR,
            backend="local",
        )
        self._vlm_fn = make_vlm_fn(
            provider="ollama",
            base_url=cfg.base_url,
            model=cfg.vlm_model,
            run_logger=run_logger,
        )
        self._caught_up_cursor: int = -1
        self._journal_buffer: list[str] = []  # accumulated journal summaries (rolling window)
        self._state_context: list[str] = []  # state context from full warm-up
        self._max_journal_lines: int = 40  # cap to avoid overflowing local model context

    @property
    def backend_type(self) -> str:
        return BackendType.LOCAL

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        return await self._agent.decide(snap, operator_cmd=operator_cmd, epoch=epoch)

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        return await self._vlm_fn(arm_id, image, known_handles, target_handle=target_handle)

    async def health_check(self) -> bool:
        """Ping Ollama /api/tags endpoint."""
        try:
            result = await asyncio.to_thread(self._ping_ollama)
            return result
        except Exception:
            return False

    def _ping_ollama(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self._config.base_url}/api/tags", timeout=3) as resp:
                data = json.loads(resp.read())
            return bool(data.get("models"))
        except Exception:
            return False

    @property
    def last_reasoning(self) -> str:
        return self._agent.last_reasoning

    def reset_loop_state(self) -> None:
        self._agent.reset_loop_state()

    # -- WarmableBackend --

    async def warm_up(
        self,
        state: CognitiveState | None,
        journal_entries: list[ContextEntry],
    ) -> bool:
        """Inject handoff context into the in-process PlannerAgent.

        Full warm-up (state provided): resets session and rebuilds context
        from scratch. Incremental (state=None): accumulates journal entries
        across batches and re-injects the full buffer without losing earlier
        batches.
        """
        if state is None and not journal_entries:
            return True

        if state is not None:
            # Full warm-up: reset everything and rebuild state context
            self._state_context = ["[Context handoff from previous backend]"]
            self._journal_buffer = []
            if state.last_scene_description:
                self._state_context.append(f"Last scene: {state.last_scene_description}")
            if state.known_scene_handles:
                self._state_context.append(f"Known objects: {', '.join(state.known_scene_handles)}")
            if state.active_target_handle:
                self._state_context.append(f"Active target: {state.active_target_handle}")
            if state.held_object_handle:
                self._state_context.append(f"Holding: {state.held_object_handle}")
            if state.recent_decisions:
                self._state_context.append("Recent decisions:")
                for d in state.recent_decisions:
                    self._state_context.append(f"  - {d}")
            if state.pending_operator_instruction:
                self._state_context.append(f"Pending operator: {state.pending_operator_instruction}")

        # Accumulate journal entries across batches (rolling window)
        if journal_entries:
            for e in journal_entries:
                self._journal_buffer.append(f"  [{e.entry_type}] {e.summary}")
            # Trim oldest entries to stay within context budget
            if len(self._journal_buffer) > self._max_journal_lines:
                self._journal_buffer = self._journal_buffer[-self._max_journal_lines :]
            self._caught_up_cursor = max(e.cursor for e in journal_entries)

        # Build full context and inject
        context_parts = list(self._state_context)
        if self._journal_buffer:
            context_parts.append("Recent journal:")
            context_parts.extend(self._journal_buffer)

        await self._agent.reset_session()
        await self._agent.inject_handoff_context("\n".join(context_parts))
        return True

    @property
    def readiness(self) -> str:
        return BackendReadiness.READY

    @property
    def caught_up_cursor(self) -> int:
        return self._caught_up_cursor
