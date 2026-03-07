"""Unit tests for SessionManager — per-arm session management with sync protocol."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from halo.cognitive.compactor import MessageHistory

from cloud_service.session_manager import ArmSession, SessionManager


def _mock_agent() -> MagicMock:
    """Create a mock PlannerAgent with all required attributes."""
    agent = MagicMock(
        decide=MagicMock(),
        last_reasoning="",
        reset_loop_state=MagicMock(),
        reset_session=AsyncMock(),
        inject_compaction_state=AsyncMock(),
    )
    agent.msg_history = MessageHistory()
    return agent


def _make_firestore_doc(**overrides) -> dict:
    """Create a Firestore session document with sensible defaults."""
    doc = {
        "client_session_id": None,
        "readiness": "ready",
        "cursor": 0,
        "pending_handoff": None,
        "active_target_handle": None,
        "held_object_handle": None,
        "known_scene_handles": [],
        "last_scene_description": "",
        "pending_operator_instruction": None,
        "next_cursor": 1,
        "entries": [],
        "last_msg_id": None,
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create minimal prompts directory."""
    d = tmp_path / "planner"
    d.mkdir()
    (d / "system_prompt.md").write_text("You are a planner.")
    return d


@pytest.fixture
def mock_firestore_store() -> MagicMock:
    store = MagicMock()
    store.save = AsyncMock()
    store.load = AsyncMock(return_value=None)
    store.delete = AsyncMock()
    return store


@pytest.fixture
def mgr(prompts_dir: Path, mock_firestore_store: MagicMock) -> SessionManager:
    """SessionManager with mocked PlannerAgent creation."""
    with patch("cloud_service.session_manager.PlannerAgent") as mock_cls:
        mock_cls.return_value = _mock_agent()
        yield SessionManager(
            model_name="test-model",
            prompts_dir=prompts_dir,
            vlm_fn_factory=lambda: MagicMock(),
            max_sessions=4,
            idle_timeout_s=0.001,  # very short for testing
            firestore_store=mock_firestore_store,
        )


@pytest.mark.asyncio
async def test_get_or_create_new_session(mgr: SessionManager):
    session = await mgr.get_or_create("arm0")
    assert isinstance(session, ArmSession)
    assert session.arm_id == "arm0"
    assert mgr.session_count == 1


@pytest.mark.asyncio
async def test_get_or_create_returns_existing(mgr: SessionManager):
    s1 = await mgr.get_or_create("arm0")
    s2 = await mgr.get_or_create("arm0")
    assert s1 is s2
    assert mgr.session_count == 1


@pytest.mark.asyncio
async def test_multiple_sessions(mgr: SessionManager):
    await mgr.get_or_create("arm0")
    await mgr.get_or_create("arm1")
    await mgr.get_or_create("arm2")
    assert mgr.session_count == 3


@pytest.mark.asyncio
async def test_eviction_at_capacity(prompts_dir: Path, mock_firestore_store: MagicMock):
    """When at max_sessions, LRU session is evicted."""
    with patch("cloud_service.session_manager.PlannerAgent") as mock_cls:
        mock_cls.return_value = _mock_agent()
        # Use a long idle timeout so idle eviction doesn't interfere
        mgr = SessionManager(
            model_name="test-model",
            prompts_dir=prompts_dir,
            vlm_fn_factory=lambda: MagicMock(),
            max_sessions=4,
            idle_timeout_s=600.0,
            firestore_store=mock_firestore_store,
        )
        await mgr.get_or_create("arm0")
        await mgr.get_or_create("arm1")
        await mgr.get_or_create("arm2")
        await mgr.get_or_create("arm3")
        assert mgr.session_count == 4
        # At capacity (4). Creating arm4 should evict arm0 (LRU)
        await mgr.get_or_create("arm4")
        assert mgr.session_count == 4
        assert mgr.get_session("arm0") is None
        assert mgr.get_session("arm4") is not None


@pytest.mark.asyncio
async def test_reset_session(mgr: SessionManager):
    await mgr.get_or_create("arm0")
    await mgr.reset_session("arm0")
    session = mgr.get_session("arm0")
    assert session is not None
    assert session.readiness == "cold"
    assert session.cursor == -1


def test_get_session_nonexistent(mgr: SessionManager):
    assert mgr.get_session("nonexistent") is None


def test_vlm_fn_shared(mgr: SessionManager):
    """VLM function is created once and shared."""
    fn1 = mgr.vlm_fn
    fn2 = mgr.vlm_fn
    assert fn1 is fn2


# ---------------------------------------------------------------------------
# sync_session protocol tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_session_fresh_when_no_last_msg_id(mgr: SessionManager):
    """last_msg_id=None creates a fresh session."""
    result = await mgr.sync_session("arm0", last_msg_id=None, msg_history=None)
    assert result.status == "ok"
    assert result.session is not None
    assert result.session.arm_id == "arm0"


@pytest.mark.asyncio
async def test_sync_session_in_memory_match(mgr: SessionManager):
    """When in-memory session's last msg_id matches, return it directly."""
    # Create session and add a message to history
    session = await mgr.get_or_create("arm0")
    msg_id = session.agent.msg_history.append("user", "hello")

    result = await mgr.sync_session("arm0", last_msg_id=msg_id, msg_history=None)
    assert result.status == "ok"
    assert result.session is session


@pytest.mark.asyncio
async def test_sync_session_need_history_when_no_match(mgr: SessionManager):
    """When nothing matches, return need_history."""
    result = await mgr.sync_session("arm0", last_msg_id="nonexistent-id", msg_history=None)
    assert result.status == "need_history"
    assert result.session is None


@pytest.mark.asyncio
async def test_sync_session_rebuild_from_history(mgr: SessionManager):
    """When msg_history is provided, rebuild session from it."""
    history = [
        {"msg_id": "m1", "role": "user", "text": "Pick the red cube", "ts_ms": 100, "is_summary": False},
        {"msg_id": "m2", "role": "model", "text": "Starting pick skill", "ts_ms": 200, "is_summary": False},
    ]
    result = await mgr.sync_session("arm0", last_msg_id="m1", msg_history=history)
    assert result.status == "ok"
    assert result.session is not None
    # inject_compaction_state should have been called
    result.session.agent.inject_compaction_state.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_session_firestore_match(mgr: SessionManager, mock_firestore_store: MagicMock):
    """When Firestore doc's last_msg_id matches, rehydrate from it."""
    mock_firestore_store.load = AsyncMock(
        return_value=_make_firestore_doc(
            last_msg_id="msg-abc",
            cursor=3,
            msg_history=[
                {"msg_id": "msg-abc", "role": "user", "text": "hello", "ts_ms": 100, "is_summary": False},
            ],
        )
    )

    result = await mgr.sync_session("arm0", last_msg_id="msg-abc", msg_history=None)
    assert result.status == "ok"
    assert result.session is not None


@pytest.mark.asyncio
async def test_sync_session_client_mismatch_creates_fresh(mgr: SessionManager):
    """Client session mismatch creates a fresh session."""
    session = await mgr.get_or_create("arm0", client_session_id="old-client")
    session.agent.msg_history.append("user", "hello")

    result = await mgr.sync_session("arm0", last_msg_id=None, msg_history=None, client_session_id="new-client")
    assert result.status == "ok"
    assert result.session is not None


# ---------------------------------------------------------------------------
# Firestore rehydration (via get_or_create)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_create_rehydrates_from_firestore(mgr: SessionManager, mock_firestore_store: MagicMock):
    """Cache miss with Firestore data + msg_history → rehydrated session with conversation replay."""
    _base = {"epoch": 1, "backend": "local", "data": {}}
    _entries = [
        {"cursor": 0, "ts_ms": 10, "entry_type": "decision", "summary": "d0", **_base},
        {
            "cursor": 1,
            "ts_ms": 20,
            "entry_type": "scene",
            "summary": "table with cube",
            "data": {"handles": ["cube"]},
            "epoch": 1,
            "backend": "local",
        },
    ]
    _history = [
        {"msg_id": "m1", "role": "user", "text": "Pick the red cube", "ts_ms": 100, "is_summary": False},
        {"msg_id": "m2", "role": "model", "text": "Starting pick skill", "ts_ms": 200, "is_summary": False},
    ]
    mock_firestore_store.load = AsyncMock(
        return_value=_make_firestore_doc(
            client_session_id="client-abc",
            cursor=3,
            active_target_handle="cube",
            known_scene_handles=["cube", "bowl"],
            last_scene_description="table with cube and bowl",
            next_cursor=4,
            entries=_entries,
            msg_history=_history,
        )
    )

    session = await mgr.get_or_create("arm0", client_session_id="client-abc")
    assert session.readiness == "ready"
    assert session.cursor == 3
    # With msg_history present, handoff is None (agent has full conversation)
    assert session.pending_handoff is None
    assert session.context_store._active_target_handle == "cube"
    assert session.context_store._known_scene_handles == ["cube", "bowl"]
    assert len(session.context_store) == 2
    # Verify inject_compaction_state was called with empty summary + retained records
    session.agent.inject_compaction_state.assert_awaited_once()
    call_args = session.agent.inject_compaction_state.call_args[0]
    assert call_args[0] == ""  # no summary
    assert len(call_args[1]) == 2  # 2 retained records


@pytest.mark.asyncio
async def test_reset_deletes_from_firestore(mgr: SessionManager, mock_firestore_store: MagicMock):
    """reset_session calls Firestore delete."""
    await mgr.get_or_create("arm0")
    await mgr.reset_session("arm0")
    mock_firestore_store.delete.assert_called_once_with("arm0")


@pytest.mark.asyncio
async def test_rehydrate_preserves_persisted_handoff(mgr: SessionManager, mock_firestore_store: MagicMock):
    """When no msg_history, rehydration uses persisted pending_handoff (richer than ContextStore)."""
    rich_handoff = (
        "[Context handoff from previous backend]\n"
        "Goal summary: Pick bowl, then place cube in bin\n"
        "Recent decisions:\n  - Pick the bowl first"
    )
    mock_firestore_store.load = AsyncMock(
        return_value=_make_firestore_doc(
            client_session_id="client-abc",
            cursor=1,
            pending_handoff=rich_handoff,
        )
    )

    session = await mgr.get_or_create("arm0", client_session_id="client-abc")
    # Persisted handoff should be preserved, not regenerated from empty ContextStore
    assert session.pending_handoff == rich_handoff
    assert "Goal summary" in session.pending_handoff


@pytest.mark.asyncio
async def test_rehydrate_replays_compacted_msg_history(mgr: SessionManager, mock_firestore_store: MagicMock):
    """Rehydration with compaction summary splits records correctly for inject_compaction_state."""
    _history = [
        {
            "msg_id": "summary-1",
            "role": "user",
            "text": "Compacted: picked cube, placed on shelf",
            "ts_ms": 50,
            "is_summary": True,
        },
        {"msg_id": "m10", "role": "user", "text": "Now pick the bowl", "ts_ms": 300, "is_summary": False},
        {"msg_id": "m11", "role": "model", "text": "Starting pick for bowl", "ts_ms": 400, "is_summary": False},
    ]
    mock_firestore_store.load = AsyncMock(
        return_value=_make_firestore_doc(
            client_session_id="client-abc",
            cursor=5,
            next_cursor=6,
            msg_history=_history,
        )
    )

    session = await mgr.get_or_create("arm0", client_session_id="client-abc")
    assert session.pending_handoff is None  # agent has full conversation
    session.agent.inject_compaction_state.assert_awaited_once()
    summary, retained = session.agent.inject_compaction_state.call_args[0]
    assert summary == "Compacted: picked cube, placed on shelf"
    assert len(retained) == 2
    assert retained[0].msg_id == "m10"
    assert retained[1].msg_id == "m11"


@pytest.mark.asyncio
async def test_rehydrate_resets_agent_on_client_mismatch(mgr: SessionManager, mock_firestore_store: MagicMock):
    """Rehydration with different client_session_id resets the agent's ADK session."""
    _history = [{"msg_id": "m1", "role": "user", "text": "old conversation", "ts_ms": 100, "is_summary": False}]
    mock_firestore_store.load = AsyncMock(
        return_value=_make_firestore_doc(
            client_session_id="old-client",
            cursor=5,
            active_target_handle="cube",
            known_scene_handles=["cube"],
            last_scene_description="table",
            next_cursor=6,
            msg_history=_history,
        )
    )

    session = await mgr.get_or_create("arm0", client_session_id="new-client")
    # Stale doc skipped — fresh session created instead
    assert session.client_session_id == "new-client"
    assert session.readiness == "cold"
    assert session.cursor == -1
    assert len(session.context_store) == 0
    assert session.pending_handoff is None


@pytest.mark.asyncio
async def test_rehydration_respects_max_sessions(prompts_dir: Path, mock_firestore_store: MagicMock):
    """Firestore rehydration evicts idle/LRU sessions when at capacity."""
    with patch("cloud_service.session_manager.PlannerAgent") as mock_cls:
        mock_cls.return_value = _mock_agent()
        mgr = SessionManager(
            model_name="test-model",
            prompts_dir=prompts_dir,
            vlm_fn_factory=lambda: MagicMock(),
            max_sessions=2,
            idle_timeout_s=600.0,
            firestore_store=mock_firestore_store,
        )
        # Fill to capacity
        await mgr.get_or_create("arm0")
        await mgr.get_or_create("arm1")
        assert mgr.session_count == 2

        # Rehydrate a third arm from Firestore — should evict LRU first
        _entry = {
            "cursor": 0,
            "ts_ms": 10,
            "epoch": 1,
            "backend": "local",
            "entry_type": "decision",
            "summary": "d0",
            "data": {},
        }
        mock_firestore_store.load = AsyncMock(return_value=_make_firestore_doc(entries=[_entry]))
        await mgr.get_or_create("arm2")
        assert mgr.session_count == 2
        assert mgr.get_session("arm0") is None  # LRU evicted
        assert mgr.get_session("arm2") is not None
