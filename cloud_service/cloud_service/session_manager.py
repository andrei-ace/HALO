"""SessionManager — per-arm_id PlannerAgent sessions with warm-up and idle eviction."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from halo.cognitive.config import BackendReadiness, CompactionConfig
from halo.cognitive.context_store import ContextEntry, ContextStore
from halo.contracts.serde import cognitive_state_from_dict, context_entry_from_dict
from halo.services.planner_service.agent import PlannerAgent

logger = logging.getLogger(__name__)


@dataclass
class ArmSession:
    """Per-arm PlannerAgent session with warm-up state tracking."""

    arm_id: str
    agent: PlannerAgent
    context_store: ContextStore = field(default_factory=ContextStore)
    cursor: int = -1
    readiness: str = BackendReadiness.COLD
    last_active_ms: int = 0
    pending_handoff: str | None = None
    client_session_id: str | None = None

    def touch(self) -> None:
        self.last_active_ms = int(time.monotonic() * 1000)


VlmFn = Callable  # relaxed type for vlm_fn


class SessionManager:
    """Manages per-arm_id PlannerAgent sessions with idle eviction."""

    def __init__(
        self,
        model_name: str,
        prompts_dir: Path,
        vlm_fn_factory: Callable[[], VlmFn],
        backend: str = "cloud",
        ollama_base_url: str = "http://localhost:11434",
        max_sessions: int = 16,
        idle_timeout_s: float = 600.0,
    ) -> None:
        self._model_name = model_name
        self._prompts_dir = prompts_dir
        self._vlm_fn_factory = vlm_fn_factory
        self._backend = backend
        self._ollama_base_url = ollama_base_url
        self._max_sessions = max_sessions
        self._idle_timeout_ms = int(idle_timeout_s * 1000)
        self._sessions: dict[str, ArmSession] = {}
        self._vlm_fn: VlmFn | None = None  # shared VLM fn (created once)
        self._nonce: str = uuid.uuid4().hex  # unique per process lifetime

    @property
    def vlm_fn(self) -> VlmFn:
        if self._vlm_fn is None:
            self._vlm_fn = self._vlm_fn_factory()
        return self._vlm_fn

    def get_or_create(self, arm_id: str, client_session_id: str | None = None) -> ArmSession:
        """Get existing session or create a new one for the given arm_id.

        If *client_session_id* is provided and differs from the stored one,
        the session is auto-reset (new TUI client connected).
        """
        if arm_id in self._sessions:
            session = self._sessions[arm_id]
            if client_session_id and session.client_session_id and client_session_id != session.client_session_id:
                old_sid = session.client_session_id[:8]
                new_sid = client_session_id[:8]
                logger.info("New client session for arm_id=%s (%s → %s), resetting", arm_id, old_sid, new_sid)
                self.reset_session(arm_id)
                session = self._sessions.get(arm_id)  # reset_session keeps the session
                if session is not None:
                    session.client_session_id = client_session_id
                    session.touch()
                    return session
                # fell through — create new below
            else:
                if client_session_id:
                    session.client_session_id = client_session_id
                session.touch()
                return session

        self._evict_if_needed()

        agent = PlannerAgent(
            model_name=self._model_name,
            base_url=self._ollama_base_url,
            prompts_dir=self._prompts_dir,
            backend=self._backend,
            compaction_config=CompactionConfig() if self._backend == "cloud" else None,
        )
        session = ArmSession(arm_id=arm_id, agent=agent, client_session_id=client_session_id)
        session.touch()
        self._sessions[arm_id] = session
        logger.info("Created new session for arm_id=%s (total=%d)", arm_id, len(self._sessions))
        return session

    def warm_up_session(
        self,
        arm_id: str,
        state_dict: dict | None,
        journal_dicts: list[dict],
        client_session_id: str | None = None,
    ) -> ArmSession:
        """Warm up a session with CognitiveState and journal entries.

        Returns the session with updated readiness and cursor.
        """
        session = self.get_or_create(arm_id, client_session_id=client_session_id)

        # Apply journal entries to the session's context store
        entries: list[ContextEntry] = []
        for jd in journal_dicts:
            entry = context_entry_from_dict(jd)
            entries.append(entry)

        if entries:
            # Filter to entries not yet applied
            new_entries = [e for e in entries if e.cursor > session.cursor]
            if new_entries:
                try:
                    session.context_store.apply_entries(new_entries)
                    session.cursor = session.context_store.latest_cursor
                except ValueError:
                    logger.warning("Cursor monotonicity issue for arm_id=%s, resetting", arm_id)
                    # Reset context store on cursor issues
                    session.context_store = ContextStore()
                    session.context_store.apply_entries(entries)
                    session.cursor = session.context_store.latest_cursor

        # If state was provided, apply tracked state
        epoch = 0
        if state_dict is not None:
            state = cognitive_state_from_dict(state_dict)
            session.context_store.set_active_target(state.active_target_handle)
            session.context_store.set_held_object(state.held_object_handle)
            epoch = state.epoch

        # Build handoff text — consumed once on next /decide
        session.pending_handoff = session.context_store.get_handoff_context(epoch)
        session.readiness = BackendReadiness.READY
        session.touch()
        return session

    def get_session(self, arm_id: str) -> ArmSession | None:
        """Get session without creating a new one."""
        return self._sessions.get(arm_id)

    def reset_session(self, arm_id: str) -> None:
        """Reset a specific arm session."""
        session = self._sessions.get(arm_id)
        if session is not None:
            session.agent.reset_loop_state()
            session.readiness = BackendReadiness.COLD
            session.cursor = -1
            session.context_store = ContextStore()
            session.pending_handoff = None
            logger.info("Reset session for arm_id=%s", arm_id)

    @property
    def nonce(self) -> str:
        """Unique identifier for this process lifetime. Changes on restart."""
        return self._nonce

    @property
    def active_arm_ids(self) -> list[str]:
        """List of arm IDs with active sessions."""
        return list(self._sessions.keys())

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def _evict_if_needed(self) -> None:
        """Evict idle sessions if at capacity."""
        if len(self._sessions) < self._max_sessions:
            return

        now_ms = int(time.monotonic() * 1000)

        # First, try to evict idle sessions
        idle_ids = [
            arm_id
            for arm_id, session in self._sessions.items()
            if (now_ms - session.last_active_ms) > self._idle_timeout_ms
        ]
        for arm_id in idle_ids:
            del self._sessions[arm_id]
            logger.info("Evicted idle session: arm_id=%s", arm_id)

        # If still at capacity, evict LRU
        if len(self._sessions) >= self._max_sessions:
            lru_id = min(self._sessions, key=lambda k: self._sessions[k].last_active_ms)
            del self._sessions[lru_id]
            logger.info("Evicted LRU session: arm_id=%s", lru_id)
