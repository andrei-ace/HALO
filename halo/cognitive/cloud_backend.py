"""Cloud cognitive backend — Gemini Live API (bidirectional audio + text).

Uses a persistent ``LivePlannerSession`` for the planner (brain) and the
standard Gemini VLM API for scene analysis (eyes).  The Live API does not
support JSON schema output, so VLM stays on the standard request-response API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from halo.cognitive.config import BackendReadiness, BackendType, CloudConfig
from halo.cognitive.context_store import CognitiveState, ContextEntry
from halo.cognitive.live_session import LivePlannerSession, LiveSessionState
from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.target_perception_service.vlm_parser import VlmScene

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS_DIR = Path(__file__).parents[2] / "configs" / "planner"


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
    """Brain (Live API planner) + eyes (standard Gemini VLM) backend."""

    def __init__(
        self,
        config: CloudConfig | None = None,
        prompts_dir: Path | None = None,
        audio_capture: object | None = None,
        audio_playback: object | None = None,
        run_logger: RunLogger | None = None,
    ) -> None:
        cfg = config or CloudConfig()
        self._config = cfg
        self._session = LivePlannerSession(
            config=cfg,
            prompts_dir=prompts_dir or _DEFAULT_PROMPTS_DIR,
            audio_capture=audio_capture,
            audio_playback=audio_playback,
        )
        # VLM stays on standard API (Live API doesn't support structured JSON schema)
        from halo.services.target_perception_service.gemini_vlm_fn import make_gemini_vlm_fn

        self._vlm_fn = make_gemini_vlm_fn(
            model=cfg.vlm_model,
            run_logger=run_logger,
        )
        self._caught_up_cursor: int = -1

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    @property
    def session_state(self) -> LiveSessionState:
        return self._session.state

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        return await self._session.decide(snap, operator_cmd=operator_cmd, epoch=epoch)

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        return await self._vlm_fn(arm_id, image, known_handles, target_handle=target_handle)

    async def health_check(self) -> bool:
        return self._session.state.connected

    @property
    def last_reasoning(self) -> str:
        return self._session.last_reasoning

    def reset_loop_state(self) -> None:
        self._session.reset_loop_state()

    # -- WarmableBackend --

    async def warm_up(
        self,
        state: CognitiveState | None,
        journal_entries: list[ContextEntry],
    ) -> bool:
        """Start the live session and inject handoff context."""
        try:
            await self._session.start()
            if state is not None and self._session.state.connected:
                context_text = _build_handoff_text(state)
                self._session.inject_handoff_context(context_text)
            if journal_entries:
                self._caught_up_cursor = max(e.cursor for e in journal_entries)
            return self._session.state.connected
        except Exception:
            logger.exception("Cloud session warm_up failed")
            return False

    @property
    def readiness(self) -> str:
        if self._session.state.connected:
            return BackendReadiness.READY
        return BackendReadiness.COLD

    @property
    def caught_up_cursor(self) -> int:
        return self._caught_up_cursor

    async def aclose(self) -> None:
        await self._session.stop()

    def drain_pending_commands(self) -> list[CommandEnvelope]:
        """Drain voice-triggered commands accumulated between ticks."""
        return self._session.drain_pending_commands()
