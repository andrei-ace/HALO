"""Cloud cognitive backend — Gemini API (standard request-response).

Uses ``PlannerAgent(backend="cloud")`` for the planner (brain) and the
Gemini VLM API for scene analysis (eyes).  Both use the standard
request-response API via ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from halo.cognitive.compactor import CompactionResult
from halo.cognitive.config import BackendReadiness, BackendType, CloudConfig, CompactionConfig
from halo.cognitive.context_store import CognitiveState, ContextEntry
from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.planner_service.agent import PlannerAgent
from halo.services.target_perception_service.vlm_parser import VlmScene

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS_DIR = Path(__file__).parents[2] / "configs" / "planner"
_QUARANTINE_DURATION_S = 300.0  # 5 minutes after a 429/quota error
_QUARANTINE_KEYWORDS = ("429", "resource_exhausted", "quota")


def _build_handoff_text(state: CognitiveState) -> str:
    """Build handoff context text from CognitiveState (same format as local_backend)."""
    parts = ["[Context handoff from previous backend]"]
    if state.last_scene_description:
        parts.append(f"Last scene: {state.last_scene_description}")
    if state.known_scene_handles:
        parts.append(f"Known objects: {', '.join(state.known_scene_handles)}")
    if state.active_target_handle:
        parts.append(f"Active target: {state.active_target_handle}")
    if state.held_object_handle:
        parts.append(f"Holding: {state.held_object_handle}")
    if state.recent_decisions:
        parts.append("Recent decisions:")
        for d in state.recent_decisions:
            parts.append(f"  - {d}")
    if state.pending_operator_instruction:
        parts.append(f"Pending operator: {state.pending_operator_instruction}")
    return "\n".join(parts)


class CloudCognitiveBackend:
    """Brain (PlannerAgent → Gemini) + eyes (Gemini VLM) backend."""

    def __init__(
        self,
        config: CloudConfig | None = None,
        prompts_dir: Path | None = None,
        run_logger: RunLogger | None = None,
        compaction_config: CompactionConfig | None = None,
    ) -> None:
        cfg = config or CloudConfig()
        self._config = cfg
        self._prompts_dir = prompts_dir or _DEFAULT_PROMPTS_DIR

        # Planner: standard PlannerAgent routed to Gemini via ADK
        self._agent = PlannerAgent(
            model_name=cfg.planner_model,
            base_url="",  # unused for cloud backend
            prompts_dir=self._prompts_dir,
            backend="cloud",
            compaction_config=compaction_config,
        )
        self._ready = False
        self._on_compaction: Callable[[CompactionResult], Awaitable[None]] | None = None

        # VLM: standard Gemini API
        from halo.services.target_perception_service.vlm_fn import make_vlm_fn

        self._vlm_fn = make_vlm_fn(
            provider="gemini",
            model=cfg.vlm_model,
            run_logger=run_logger,
        )
        self._caught_up_cursor: int = -1
        self._quarantine_until: float = 0.0  # monotonic time; 0 = not quarantined

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        try:
            commands = await self._agent.decide(snap, operator_cmd=operator_cmd)
            # Check for ADK compaction after successful decide
            compaction = self._agent.last_compaction
            if compaction is not None and self._on_compaction is not None:
                try:
                    await self._on_compaction(compaction)
                except Exception:
                    logger.debug("Compaction callback failed", exc_info=True)
            return commands
        except Exception as exc:
            self._maybe_quarantine(exc)
            raise

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        try:
            return await self._vlm_fn(arm_id, image, known_handles, target_handle=target_handle)
        except Exception as exc:
            self._maybe_quarantine(exc)
            raise

    def _maybe_quarantine(self, exc: Exception) -> None:
        msg = str(exc).lower()
        if any(kw in msg for kw in _QUARANTINE_KEYWORDS):
            self._quarantine_until = time.monotonic() + _QUARANTINE_DURATION_S
            logger.info("Cloud backend quarantined for %.0fs after: %s", _QUARANTINE_DURATION_S, exc)

    async def health_check(self) -> bool:
        if not os.environ.get("GOOGLE_API_KEY"):
            return False
        if time.monotonic() < self._quarantine_until:
            return False
        return True

    @property
    def model_name(self) -> str:
        return self._agent.model_name

    @property
    def last_reasoning(self) -> str:
        return self._agent.last_reasoning

    @property
    def last_token_usage(self) -> dict[str, int]:
        return self._agent.last_token_usage

    @property
    def agent(self) -> PlannerAgent:
        return self._agent

    def set_on_compaction(self, callback: Callable[[CompactionResult], Awaitable[None]] | None) -> None:
        self._on_compaction = callback

    def reset_loop_state(self) -> None:
        self._agent.reset_loop_state()
        self._ready = False
        self._caught_up_cursor = -1

    # -- WarmableBackend --

    async def warm_up(
        self,
        state: CognitiveState | None,
        journal_entries: list[ContextEntry],
    ) -> bool:
        """Mark as ready and inject handoff context if provided.

        Returns False if health_check() fails (e.g. missing API key or
        quarantined) — callers must not treat the backend as usable.
        """
        try:
            if not await self.health_check():
                return False
            if state is not None:
                context_text = _build_handoff_text(state)
                # Inject handoff as an operator message on next decide()
                self._agent._pending_handoff = context_text
            if journal_entries:
                self._caught_up_cursor = max(e.cursor for e in journal_entries)
            self._ready = True
            return True
        except Exception:
            logger.debug("Cloud warm_up failed", exc_info=True)
            return False

    @property
    def readiness(self) -> str:
        if self._ready:
            return BackendReadiness.READY
        return BackendReadiness.COLD

    @property
    def caught_up_cursor(self) -> int:
        return self._caught_up_cursor

    async def aclose(self) -> None:
        pass
