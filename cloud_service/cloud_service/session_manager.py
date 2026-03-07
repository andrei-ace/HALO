"""SessionManager — per-arm_id PlannerAgent sessions with warm-up and idle eviction."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from halo.cognitive.config import BackendReadiness, CompactionConfig
from halo.cognitive.context_store import CognitiveState, ContextEntry, ContextStore
from halo.contracts.serde import cognitive_state_from_dict, context_entry_from_dict
from halo.services.planner_service.agent import PlannerAgent

if TYPE_CHECKING:
    from cloud_service.firestore_store import FirestoreSessionStore

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
        max_sessions: int = 16,
        idle_timeout_s: float = 600.0,
        compaction_interval: int = 20,
        compaction_overlap: int = 4,
        firestore_store: FirestoreSessionStore | None = None,
    ) -> None:
        self._model_name = model_name
        self._prompts_dir = prompts_dir
        self._vlm_fn_factory = vlm_fn_factory
        self._max_sessions = max_sessions
        self._idle_timeout_ms = int(idle_timeout_s * 1000)
        self._compaction_interval = compaction_interval
        self._compaction_overlap = compaction_overlap
        self._sessions: dict[str, ArmSession] = {}
        self._vlm_fn: VlmFn | None = None  # shared VLM fn (created once)
        self._nonce: str = uuid.uuid4().hex  # unique per process lifetime
        self._firestore_store = firestore_store

    @property
    def vlm_fn(self) -> VlmFn:
        if self._vlm_fn is None:
            self._vlm_fn = self._vlm_fn_factory()
        return self._vlm_fn

    async def get_or_create(self, arm_id: str, client_session_id: str | None = None) -> ArmSession:
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
                await self.reset_session(arm_id)
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

        # Try Firestore rehydration before creating a brand-new session
        if self._firestore_store is not None:
            doc = await self._firestore_store.load(arm_id)
            if doc is not None:
                # Check client mismatch *before* expensive deserialization
                stored_sid = doc.get("client_session_id")
                if client_session_id and stored_sid and client_session_id != stored_sid:
                    logger.info(
                        "New client session on rehydrate for arm_id=%s (%s → %s), skipping stale doc",
                        arm_id,
                        stored_sid[:8],
                        client_session_id[:8],
                    )
                    # Fall through to create a fresh session below
                else:
                    session = await self._rehydrate_from_firestore(arm_id, doc)
                    if client_session_id:
                        session.client_session_id = client_session_id
                    session.touch()
                    self._sessions[arm_id] = session
                    logger.info("Rehydrated session from Firestore for arm_id=%s", arm_id)
                    return session

        agent = self._create_agent()
        session = ArmSession(arm_id=arm_id, agent=agent, client_session_id=client_session_id)
        session.touch()
        self._sessions[arm_id] = session
        logger.info("Created new session for arm_id=%s (total=%d)", arm_id, len(self._sessions))
        return session

    def _create_agent(self) -> PlannerAgent:
        return PlannerAgent(
            model_name=self._model_name,
            base_url="",  # unused — cloud backend routes via GOOGLE_API_KEY
            prompts_dir=self._prompts_dir,
            backend="cloud",
            compaction_config=CompactionConfig(
                compaction_interval=self._compaction_interval,
                overlap_size=self._compaction_overlap,
            ),
        )

    @staticmethod
    def _apply_tracked_state(
        context_store: ContextStore,
        *,
        active_target: str | None,
        held_object: str | None,
        known_handles: list[str],
        scene_desc: str,
        operator_instruction: str | None,
    ) -> None:
        """Apply tracked state fields to a ContextStore (used by both warm-up and rehydration)."""
        context_store.set_active_target(active_target)
        context_store.set_held_object(held_object)
        context_store._known_scene_handles = list(known_handles)
        context_store._last_scene_description = scene_desc
        context_store._pending_operator_instruction = operator_instruction

    @staticmethod
    def _build_handoff_from_state(state: CognitiveState) -> str:
        """Build a rich handoff from CognitiveState for empty-journal warm-ups."""
        parts = ["[Context handoff from previous backend]"]

        if state.last_scene_description:
            parts.append(f"Last scene analysis: {state.last_scene_description}")
        if state.known_scene_handles:
            parts.append(f"Known objects: {', '.join(state.known_scene_handles)}")
        if state.active_target_handle:
            parts.append(f"Active target: {state.active_target_handle}")
        if state.held_object_handle:
            parts.append(f"Currently holding: {state.held_object_handle}")
        if state.goal_summary:
            parts.append(f"Goal summary: {state.goal_summary}")
        if state.recent_decisions:
            parts.append("Recent decisions:")
            for decision in state.recent_decisions:
                parts.append(f"  - {decision}")
        if state.pending_operator_instruction:
            parts.append(f"Pending operator instruction: {state.pending_operator_instruction}")
        if state.recent_event_summaries:
            parts.append("Recent events:")
            for event_summary in state.recent_event_summaries:
                parts.append(f"  - {event_summary}")

        runtime_bits: list[str] = []
        if state.last_skill_name:
            runtime_bits.append(f"skill={state.last_skill_name}")
        if state.last_skill_phase:
            runtime_bits.append(f"phase={state.last_skill_phase}")
        if state.last_outcome_state:
            runtime_bits.append(f"outcome={state.last_outcome_state}")
        if state.last_snapshot_id:
            runtime_bits.append(f"snapshot={state.last_snapshot_id}")
        if runtime_bits:
            parts.append(f"Latest runtime: {', '.join(runtime_bits)}")

        return "\n".join(parts)

    async def _rehydrate_from_firestore(self, arm_id: str, doc: dict) -> ArmSession:
        """Rebuild an ArmSession from a Firestore document.

        Restores ContextStore, tracked state, and — when available — replays
        the persisted MessageHistory into the fresh PlannerAgent's ADK session
        so cross-instance rehydration preserves the planner conversation
        (including compaction summaries).
        """
        from cloud_service.firestore_store import _message_record_from_dict

        agent = self._create_agent()
        cs = ContextStore()

        # Restore entries
        entries = [context_entry_from_dict(ed) for ed in doc.get("entries", [])]
        if entries:
            cs.apply_entries(entries)

        # Restore tracked state (reuse shared helper)
        self._apply_tracked_state(
            cs,
            active_target=doc.get("active_target_handle"),
            held_object=doc.get("held_object_handle"),
            known_handles=doc.get("known_scene_handles", []),
            scene_desc=doc.get("last_scene_description", ""),
            operator_instruction=doc.get("pending_operator_instruction"),
        )

        # Replay persisted message history into the fresh PlannerAgent.
        # This reconstructs the ADK session so the planner continues with
        # full conversation context instead of just a handoff string.
        raw_history = doc.get("msg_history", [])
        if raw_history:
            records = [_message_record_from_dict(rd) for rd in raw_history]
            # Split into summary (if any) + retained records for inject_compaction_state
            summary: str | None = None
            retained: list = []
            for rec in records:
                if rec.is_summary:
                    summary = rec.text
                else:
                    retained.append(rec)
            if summary is not None:
                await agent.inject_compaction_state(summary, retained)
            else:
                # No compaction yet — replay the raw transcript without inventing a summary turn.
                await agent.inject_compaction_state("", retained)
            pending_handoff = None  # agent has full conversation — no handoff needed
        else:
            # No message history — prefer the persisted handoff (may be richer than
            # what ContextStore can regenerate, e.g. goal_summary, recent_decisions).
            persisted_handoff = doc.get("pending_handoff")
            if persisted_handoff:
                pending_handoff = persisted_handoff
            else:
                has_context = (
                    entries or cs._active_target_handle or cs._held_object_handle or cs._last_scene_description
                )
                pending_handoff = cs.get_handoff_context(0) if has_context else None

        return ArmSession(
            arm_id=arm_id,
            agent=agent,
            context_store=cs,
            cursor=doc.get("cursor", -1),
            readiness=doc.get("readiness", BackendReadiness.COLD),
            pending_handoff=pending_handoff,
            client_session_id=doc.get("client_session_id"),
        )

    async def persist_session(self, arm_id: str) -> None:
        """Persist current in-memory session to Firestore."""
        if self._firestore_store is not None and arm_id in self._sessions:
            await self._firestore_store.save(arm_id, self._sessions[arm_id])

    async def warm_up_session(
        self,
        arm_id: str,
        state_dict: dict | None,
        journal_dicts: list[dict],
        client_session_id: str | None = None,
    ) -> ArmSession:
        """Warm up a session with CognitiveState and journal entries.

        Returns the session with updated readiness and cursor.
        """
        session = await self.get_or_create(arm_id, client_session_id=client_session_id)

        # Apply journal entries incrementally (supports batched catch-up from Switchboard)
        entries: list[ContextEntry] = [context_entry_from_dict(jd) for jd in journal_dicts]
        changed = False
        if entries:
            new_entries = [e for e in entries if e.cursor > session.cursor]
            if new_entries:
                try:
                    session.context_store.apply_entries(new_entries)
                    session.cursor = session.context_store.latest_cursor
                except ValueError:
                    logger.warning("Cursor monotonicity issue for arm_id=%s, resetting store", arm_id)
                    session.context_store = ContextStore()
                    session.context_store.apply_entries(entries)
                    session.cursor = session.context_store.latest_cursor
                changed = True
        elif state_dict is not None and session.agent.msg_history.count() == 0:
            # Empty journal with state on a fresh agent (no Firestore history) =
            # full warm-up from a new client — clear stale context.
            # Skip when the agent already has conversation history (e.g. just
            # rehydrated from Firestore) to avoid destroying the recovered session.
            await session.agent.reset_session()
            session.context_store = ContextStore()
            session.cursor = -1
            changed = True

        # If state was provided, apply tracked state
        epoch = 0
        state: CognitiveState | None = None
        if state_dict is not None:
            state = cognitive_state_from_dict(state_dict)
            self._apply_tracked_state(
                session.context_store,
                active_target=state.active_target_handle,
                held_object=state.held_object_handle,
                known_handles=state.known_scene_handles,
                scene_desc=state.last_scene_description,
                operator_instruction=state.pending_operator_instruction,
            )
            epoch = state.epoch
            changed = True

        # Build handoff text — consumed once on next /decide.
        # If Firestore already restored planner history into msg_history, the
        # next /decide should continue from that transcript directly instead of
        # prepending a synthetic handoff that duplicates context.
        if session.agent.msg_history.count() > 0:
            session.pending_handoff = None
        elif state is not None and not entries:
            session.pending_handoff = self._build_handoff_from_state(state)
        else:
            session.pending_handoff = session.context_store.get_handoff_context(epoch)
        session.readiness = BackendReadiness.READY
        session.touch()

        if changed:
            await self.persist_session(arm_id)
        return session

    def get_session(self, arm_id: str) -> ArmSession | None:
        """Get session without creating a new one."""
        return self._sessions.get(arm_id)

    def evict_session(self, arm_id: str) -> None:
        """Remove an in-memory session without touching Firestore.

        Used to discard tainted state after a persist failure so the next
        request rehydrates from the last good Firestore snapshot.
        """
        if arm_id in self._sessions:
            del self._sessions[arm_id]
            logger.warning("Evicted tainted session for arm_id=%s", arm_id)

    async def reset_session(self, arm_id: str) -> None:
        """Reset a specific arm session (clears ADK conversation history)."""
        session = self._sessions.get(arm_id)
        if session is not None:
            await session.agent.reset_session()
            session.readiness = BackendReadiness.COLD
            session.cursor = -1
            session.context_store = ContextStore()
            session.pending_handoff = None
            logger.info("Reset session for arm_id=%s", arm_id)
        if self._firestore_store is not None:
            await self._firestore_store.delete(arm_id)

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
