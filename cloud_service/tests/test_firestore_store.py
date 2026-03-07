"""Unit tests for FirestoreSessionStore with mocked AsyncClient."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from halo.cognitive.compactor import MessageHistory
from halo.cognitive.context_store import ContextStore

from cloud_service.firestore_store import FirestoreSessionStore
from cloud_service.session_manager import ArmSession


def _make_session(arm_id: str = "arm0", cursor: int = 1) -> ArmSession:
    """Create a minimal ArmSession for testing."""
    cs = ContextStore()
    cs.append(epoch=1, backend="local", entry_type="decision", summary="d0")
    cs.append(epoch=1, backend="local", entry_type="scene", summary="table with cube", data={"handles": ["cube"]})
    cs.set_active_target("cube")

    agent = MagicMock()
    # Give the agent a MessageHistory with some records
    mh = MessageHistory()
    mh.append("user", "Pick the cube")
    mh.append("model", "Starting pick")
    agent.msg_history = mh
    return ArmSession(
        arm_id=arm_id,
        agent=agent,
        context_store=cs,
        cursor=cursor,
        readiness="ready",
        pending_handoff="handoff text",
        client_session_id="client-123",
    )


def _mock_doc(data: dict | None, exists: bool = True) -> MagicMock:
    doc = MagicMock()
    doc.exists = exists
    doc.to_dict.return_value = data
    return doc


@pytest.fixture
def mock_client():
    with patch("cloud_service.firestore_store.AsyncClient") as cls:
        # AsyncClient.collection() and .document() are sync; only .set/.get/.delete are async
        doc_ref = MagicMock()
        doc_ref.set = AsyncMock()
        doc_ref.get = AsyncMock(return_value=_mock_doc(None, exists=False))
        doc_ref.delete = AsyncMock()

        col_ref = MagicMock()
        col_ref.document.return_value = doc_ref

        client = MagicMock()
        client.collection.return_value = col_ref
        cls.return_value = client
        yield client


@pytest.fixture
def store(mock_client) -> FirestoreSessionStore:
    return FirestoreSessionStore(collection="test_sessions", ttl_hours=1.0)


@pytest.mark.asyncio
async def test_save_serializes_session(store: FirestoreSessionStore, mock_client):
    session = _make_session()
    await store.save("arm0", session)

    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.set.assert_called_once()
    saved = doc_ref.set.call_args[0][0]

    assert saved["client_session_id"] == "client-123"
    assert saved["readiness"] == "ready"
    assert saved["cursor"] == 1
    assert saved["pending_handoff"] == "handoff text"
    assert saved["active_target_handle"] == "cube"
    assert saved["known_scene_handles"] == ["cube"]
    assert len(saved["entries"]) == 2
    assert saved["entries"][0]["entry_type"] == "decision"
    assert saved["entries"][1]["entry_type"] == "scene"
    assert "updated_at" in saved
    assert "expires_at" in saved
    # Verify msg_history is serialized
    assert "msg_history" in saved
    assert len(saved["msg_history"]) == 2
    assert saved["msg_history"][0]["role"] == "user"
    assert saved["msg_history"][0]["text"] == "Pick the cube"
    assert saved["msg_history"][1]["role"] == "model"
    assert saved["msg_history"][1]["text"] == "Starting pick"
    assert saved["msg_history"][1]["is_summary"] is False


@pytest.mark.asyncio
async def test_load_returns_none_for_missing(store: FirestoreSessionStore, mock_client):
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=_mock_doc(None, exists=False))

    result = await store.load("arm0")
    assert result is None


@pytest.mark.asyncio
async def test_load_returns_none_for_expired(store: FirestoreSessionStore, mock_client):
    expired_data = {
        "cursor": 1,
        "expires_at": datetime.now(UTC) - timedelta(hours=2),
    }
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=_mock_doc(expired_data))

    result = await store.load("arm0")
    assert result is None


@pytest.mark.asyncio
async def test_load_returns_dict_for_valid(store: FirestoreSessionStore, mock_client):
    valid_data = {
        "cursor": 5,
        "readiness": "ready",
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
        "entries": [],
    }
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=_mock_doc(valid_data))

    result = await store.load("arm0")
    assert result is not None
    assert result["cursor"] == 5


@pytest.mark.asyncio
async def test_delete_removes_document(store: FirestoreSessionStore, mock_client):
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.delete = AsyncMock()

    await store.delete("arm0")
    doc_ref.delete.assert_called_once()


@pytest.mark.asyncio
async def test_roundtrip_entries(store: FirestoreSessionStore, mock_client):
    """Save then verify entry structure matches what context_entry_from_dict expects."""
    session = _make_session()
    await store.save("arm0", session)

    doc_ref = mock_client.collection.return_value.document.return_value
    saved = doc_ref.set.call_args[0][0]

    # Verify entries are valid dicts with required keys
    for entry_dict in saved["entries"]:
        assert "cursor" in entry_dict
        assert "ts_ms" in entry_dict
        assert "epoch" in entry_dict
        assert "backend" in entry_dict
        assert "entry_type" in entry_dict
        assert "summary" in entry_dict

    # Verify they can be deserialized
    from halo.contracts.serde import context_entry_from_dict

    entries = [context_entry_from_dict(ed) for ed in saved["entries"]]
    assert len(entries) == 2
    assert entries[0].entry_type == "decision"
    assert entries[1].entry_type == "scene"
