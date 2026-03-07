"""FirestoreSessionStore — persist ArmSession state in Firestore for multi-instance Cloud Run."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from google.cloud.firestore_v1 import AsyncClient
from halo.contracts.serde import context_entry_to_dict, message_record_to_dict

if TYPE_CHECKING:
    from cloud_service.session_manager import ArmSession

logger = logging.getLogger(__name__)


class FirestoreSessionStore:
    """Persist per-arm session state in Firestore.

    One document per arm_id in the configured collection.
    Uses ADC on Cloud Run; locally use FIRESTORE_EMULATOR_HOST or ADC creds.
    """

    def __init__(self, collection: str = "halo_sessions", ttl_hours: float = 1.0) -> None:
        self._collection = collection
        self._ttl = timedelta(hours=ttl_hours)
        self._client: AsyncClient | None = None

    def _get_client(self) -> AsyncClient:
        if self._client is None:
            self._client = AsyncClient()
        return self._client

    async def save(self, arm_id: str, session: ArmSession) -> None:
        """Serialize and persist an ArmSession to Firestore."""
        now = datetime.now(UTC)
        cs = session.context_store

        all_records = session.agent.msg_history.get_all()
        last_msg_id = all_records[-1].msg_id if all_records else None

        doc = {
            "client_session_id": session.client_session_id,
            "readiness": session.readiness,
            "cursor": session.cursor,
            "pending_handoff": session.pending_handoff,
            "updated_at": now,
            "expires_at": now + self._ttl,
            # Fast sync check — avoids deserializing full history
            "last_msg_id": last_msg_id,
            # ContextStore state
            "active_target_handle": cs._active_target_handle,
            "held_object_handle": cs._held_object_handle,
            "known_scene_handles": list(cs._known_scene_handles),
            "last_scene_description": cs._last_scene_description,
            "pending_operator_instruction": cs._pending_operator_instruction,
            "next_cursor": cs._next_cursor,
            "entries": [context_entry_to_dict(e) for e in cs._entries],
            # PlannerAgent conversation history (for ADK session reconstruction)
            "msg_history": [message_record_to_dict(r) for r in all_records],
        }

        client = self._get_client()
        await client.collection(self._collection).document(arm_id).set(doc)
        logger.debug("Saved session to Firestore: arm_id=%s, cursor=%d", arm_id, session.cursor)

    async def load(self, arm_id: str) -> dict | None:
        """Load session state from Firestore. Returns None if missing or expired."""
        client = self._get_client()
        doc_ref = client.collection(self._collection).document(arm_id)
        doc = await doc_ref.get()

        if not doc.exists:
            return None

        data = doc.to_dict()
        expires_at = data.get("expires_at")
        if expires_at is not None:
            # Firestore returns timezone-aware datetimes
            if isinstance(expires_at, datetime):
                now = datetime.now(UTC)
                if expires_at < now:
                    logger.info("Expired Firestore session for arm_id=%s", arm_id)
                    return None

        logger.debug("Loaded session from Firestore: arm_id=%s, cursor=%d", arm_id, data.get("cursor", -1))
        return data

    async def delete(self, arm_id: str) -> None:
        """Delete a session document from Firestore."""
        client = self._get_client()
        await client.collection(self._collection).document(arm_id).delete()
        logger.debug("Deleted Firestore session: arm_id=%s", arm_id)
