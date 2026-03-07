"""SessionManager — per-arm_id PlannerAgent sessions with sync protocol and idle eviction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from halo.cognitive.compactor import MessageRecord
from halo.cognitive.config import BackendReadiness, CompactionConfig
from halo.cognitive.context_store import ContextStore
from halo.contracts.serde import context_entry_from_dict, message_record_from_dict
from halo.services.planner_service.agent import PlannerAgent

if TYPE_CHECKING:
    from cloud_service.firestore_store import FirestoreSessionStore

logger = logging.getLogger(__name__)


@dataclass
class ArmSession:
    """Per-arm PlannerAgent session."""

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


@dataclass
class SyncResult:
    """Outcome of sync_session()."""

    status: str  # "ok" | "need_history"
    session: ArmSession | None = None


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
        self._firestore_store = firestore_store

    @property
    def vlm_fn(self) -> VlmFn:
        if self._vlm_fn is None:
            self._vlm_fn = self._vlm_fn_factory()
        return self._vlm_fn

    async def sync_session(
        self,
        arm_id: str,
        last_msg_id: str | None,
        msg_history: list[dict] | None,
        client_session_id: str | None = None,
    ) -> SyncResult:
        """Sync session using last_msg_id protocol.

        Flow:
        1. msg_history provided → rebuild session from it, persist to Firestore
        2. last_msg_id is None → create fresh session
        3. in-memory session matches → proceed (happy path)
        4. Firestore doc matches → rehydrate
        5. nothing matches → return {status: "need_history"}
        """
        # 1. msg_history provided → rebuild session
        if msg_history is not None:
            session = await self._rebuild_from_history(arm_id, msg_history, client_session_id)
            return SyncResult(status="ok", session=session)

        # 2. last_msg_id is None → fresh session
        if last_msg_id is None:
            session = await self._create_fresh(arm_id, client_session_id)
            return SyncResult(status="ok", session=session)

        # 3. in-memory session matches?
        if arm_id in self._sessions:
            session = self._sessions[arm_id]
            # Check client mismatch
            if client_session_id and session.client_session_id and client_session_id != session.client_session_id:
                logger.info("Client session mismatch for arm_id=%s, creating fresh", arm_id)
                session = await self._create_fresh(arm_id, client_session_id)
                return SyncResult(status="ok", session=session)
            if client_session_id:
                session.client_session_id = client_session_id
            # Check if last_msg_id matches
            records = session.agent.msg_history.get_all()
            if records and records[-1].msg_id == last_msg_id:
                session.touch()
                return SyncResult(status="ok", session=session)

        # 4. Try Firestore rehydration
        if self._firestore_store is not None:
            doc = await self._firestore_store.load(arm_id)
            if doc is not None:
                stored_last_msg_id = doc.get("last_msg_id")
                stored_sid = doc.get("client_session_id")
                # Client mismatch → skip stale doc
                if client_session_id and stored_sid and client_session_id != stored_sid:
                    pass  # fall through to need_history
                elif stored_last_msg_id == last_msg_id:
                    session = await self._rehydrate_from_firestore(arm_id, doc)
                    if client_session_id:
                        session.client_session_id = client_session_id
                    session.touch()
                    self._sessions[arm_id] = session
                    logger.info("Rehydrated session from Firestore for arm_id=%s", arm_id)
                    return SyncResult(status="ok", session=session)

        # 5. Nothing matches → need history
        return SyncResult(status="need_history")

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

    async def _create_fresh(self, arm_id: str, client_session_id: str | None) -> ArmSession:
        """Create a fresh session (used when last_msg_id is None)."""
        self._evict_if_needed()
        # If there's an existing session, reset it
        if arm_id in self._sessions:
            await self.reset_session(arm_id)
            session = self._sessions.get(arm_id)
            if session is not None:
                session.client_session_id = client_session_id
                session.touch()
                return session
        agent = self._create_agent()
        session = ArmSession(arm_id=arm_id, agent=agent, client_session_id=client_session_id)
        session.touch()
        self._sessions[arm_id] = session
        logger.info("Created fresh session for arm_id=%s (total=%d)", arm_id, len(self._sessions))
        return session

    async def _rebuild_from_history(
        self, arm_id: str, msg_history_dicts: list[dict], client_session_id: str | None
    ) -> ArmSession:
        """Rebuild session from client-provided msg_history."""
        self._evict_if_needed()
        agent = self._create_agent()
        records = [message_record_from_dict(d) for d in msg_history_dicts]

        # Split into summary + retained
        summary: str | None = None
        retained: list[MessageRecord] = []
        for rec in records:
            if rec.is_summary:
                summary = rec.text
            else:
                retained.append(rec)

        if summary is not None:
            await agent.inject_compaction_state(summary, retained)
        elif retained:
            await agent.inject_compaction_state("", retained)

        session = ArmSession(
            arm_id=arm_id,
            agent=agent,
            client_session_id=client_session_id,
            readiness=BackendReadiness.READY,
        )
        session.touch()
        self._sessions[arm_id] = session

        # Persist to Firestore
        await self.persist_session(arm_id)
        logger.info("Rebuilt session from client history for arm_id=%s (%d records)", arm_id, len(records))
        return session

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
        """Apply tracked state fields to a ContextStore (used by rehydration)."""
        context_store.set_active_target(active_target)
        context_store.set_held_object(held_object)
        context_store._known_scene_handles = list(known_handles)
        context_store._last_scene_description = scene_desc
        context_store._pending_operator_instruction = operator_instruction

    async def _rehydrate_from_firestore(self, arm_id: str, doc: dict) -> ArmSession:
        """Rebuild an ArmSession from a Firestore document."""
        agent = self._create_agent()
        cs = ContextStore()

        # Restore entries
        entries = [context_entry_from_dict(ed) for ed in doc.get("entries", [])]
        if entries:
            cs.apply_entries(entries)

        # Restore tracked state
        self._apply_tracked_state(
            cs,
            active_target=doc.get("active_target_handle"),
            held_object=doc.get("held_object_handle"),
            known_handles=doc.get("known_scene_handles", []),
            scene_desc=doc.get("last_scene_description", ""),
            operator_instruction=doc.get("pending_operator_instruction"),
        )

        # Replay persisted message history into the fresh PlannerAgent
        raw_history = doc.get("msg_history", [])
        if raw_history:
            records = [message_record_from_dict(rd) for rd in raw_history]
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
                await agent.inject_compaction_state("", retained)
            pending_handoff = None
        else:
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

    def get_session(self, arm_id: str) -> ArmSession | None:
        """Get session without creating a new one."""
        return self._sessions.get(arm_id)

    def evict_session(self, arm_id: str) -> None:
        """Remove an in-memory session without touching Firestore."""
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
