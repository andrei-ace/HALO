"""LiveAgentManager — per-arm Live Agent session lifecycle management."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from cloud_service.live_agent import LiveAgentSession

logger = logging.getLogger(__name__)


class LiveAgentManager:
    """Manages per-arm LiveAgentSession instances with idle eviction."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-native-audio-preview-12-2025",
        voice: str = "Kore",
        prompts_dir: Path | None = None,
        max_sessions: int = 8,
        idle_timeout_s: float = 600.0,
    ) -> None:
        self._model = model
        self._voice = voice
        self._prompts_dir = prompts_dir
        self._max_sessions = max_sessions
        self._idle_timeout_s = idle_timeout_s

        self._sessions: dict[str, LiveAgentSession] = {}
        self._last_active: dict[str, float] = {}

    @property
    def active_arm_ids(self) -> list[str]:
        return list(self._sessions.keys())

    async def get_or_create(self, arm_id: str) -> LiveAgentSession:
        """Get existing session or create a new one for the given arm_id."""
        if arm_id in self._sessions:
            self._last_active[arm_id] = time.monotonic()
            return self._sessions[arm_id]

        # Evict if at capacity
        await self._evict_if_needed()

        session = LiveAgentSession(
            arm_id=arm_id,
            model=self._model,
            voice=self._voice,
            prompts_dir=self._prompts_dir,
        )
        await session.start()

        self._sessions[arm_id] = session
        self._last_active[arm_id] = time.monotonic()
        logger.info("Created live agent session for arm_id=%s", arm_id)
        return session

    async def remove(self, arm_id: str) -> None:
        """Stop and remove a session."""
        session = self._sessions.pop(arm_id, None)
        self._last_active.pop(arm_id, None)
        if session is not None:
            await session.stop()
            logger.info("Removed live agent session for arm_id=%s", arm_id)

    async def remove_all(self) -> None:
        """Stop and remove all sessions."""
        for arm_id in list(self._sessions):
            await self.remove(arm_id)

    async def _evict_if_needed(self) -> None:
        """Evict idle or LRU sessions when at capacity."""
        if len(self._sessions) < self._max_sessions:
            return

        now = time.monotonic()

        # First pass: evict sessions idle beyond timeout
        for arm_id in list(self._sessions):
            last = self._last_active.get(arm_id, 0)
            if now - last > self._idle_timeout_s:
                await self.remove(arm_id)
                logger.info("Evicted idle live agent session arm_id=%s", arm_id)

        # Still at capacity? Evict LRU
        if len(self._sessions) >= self._max_sessions:
            lru_arm = min(self._last_active, key=self._last_active.get)  # type: ignore[arg-type]
            await self.remove(lru_arm)
            logger.info("Evicted LRU live agent session arm_id=%s", lru_arm)
