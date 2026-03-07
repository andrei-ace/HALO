"""Unit tests for SessionManager — per-arm session management."""

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
async def test_warm_up_session(mgr: SessionManager):
    state_dict = {
        "ts_ms": 100,
        "epoch": 1,
        "cursor": 2,
        "active_target_handle": "cube",
        "held_object_handle": None,
        "known_scene_handles": ["cube"],
        "last_scene_description": "table",
        "pending_operator_instruction": None,
        "recent_decisions": ["d1"],
        "last_snapshot_id": "snap-1",
        "last_arm_id": "arm0",
        "last_skill_phase": None,
        "last_skill_name": None,
        "last_outcome_state": None,
    }
    journal = [
        {
            "cursor": 0,
            "ts_ms": 10,
            "epoch": 1,
            "backend": "local",
            "entry_type": "decision",
            "summary": "d0",
            "data": {},
        },
        {
            "cursor": 1,
            "ts_ms": 20,
            "epoch": 1,
            "backend": "local",
            "entry_type": "scene",
            "summary": "table with cube",
            "data": {"handles": ["cube"]},
        },
    ]

    session = await mgr.warm_up_session("arm0", state_dict, journal)
    assert session.readiness == "ready"
    assert session.cursor == 1
    assert session.context_store._active_target_handle == "cube"


@pytest.mark.asyncio
async def test_warm_up_incremental(mgr: SessionManager):
    """Warm-up with only new journal entries (incremental)."""
    # First warm-up with 2 entries
    journal1 = [
        {
            "cursor": 0,
            "ts_ms": 10,
            "epoch": 1,
            "backend": "local",
            "entry_type": "decision",
            "summary": "d0",
            "data": {},
        },
        {
            "cursor": 1,
            "ts_ms": 20,
            "epoch": 1,
            "backend": "local",
            "entry_type": "decision",
            "summary": "d1",
            "data": {},
        },
    ]
    session = await mgr.warm_up_session("arm0", None, journal1)
    assert session.cursor == 1

    # Incremental warm-up with new entries
    journal2 = [
        {
            "cursor": 2,
            "ts_ms": 30,
            "epoch": 1,
            "backend": "local",
            "entry_type": "decision",
            "summary": "d2",
            "data": {},
        },
    ]
    session = await mgr.warm_up_session("arm0", None, journal2)
    assert session.cursor == 2


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
async def test_warm_up_persists_to_firestore(mgr: SessionManager, mock_firestore_store: MagicMock):
    """warm_up_session calls Firestore save."""
    journal = [
        {
            "cursor": 0,
            "ts_ms": 10,
            "epoch": 1,
            "backend": "local",
            "entry_type": "decision",
            "summary": "d0",
            "data": {},
        },
    ]
    await mgr.warm_up_session("arm0", None, journal)
    mock_firestore_store.save.assert_called_once()
    call_args = mock_firestore_store.save.call_args
    assert call_args[0][0] == "arm0"


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
async def test_rehydrate_regenerates_handoff_when_none(mgr: SessionManager, mock_firestore_store: MagicMock):
    """Rehydration always regenerates pending_handoff from ContextStore entries,
    even when the persisted value is None (consumed by a prior /decide)."""
    _scene_entry = {
        "cursor": 0,
        "ts_ms": 10,
        "epoch": 1,
        "backend": "local",
        "entry_type": "scene",
        "summary": "table with cube",
        "data": {"handles": ["cube"]},
    }
    mock_firestore_store.load = AsyncMock(
        return_value=_make_firestore_doc(
            client_session_id="client-abc",
            cursor=1,
            known_scene_handles=["cube"],
            last_scene_description="table with cube",
            next_cursor=2,
            entries=[_scene_entry],
        )
    )

    session = await mgr.get_or_create("arm0", client_session_id="client-abc")
    # Handoff is regenerated, not None
    assert session.pending_handoff is not None
    assert "table with cube" in session.pending_handoff


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
async def test_warm_up_full_reset_clears_agent_history(mgr: SessionManager):
    """Full warm-up (state + empty journal) resets the agent's ADK session."""
    _e = {"backend": "local", "entry_type": "decision", "data": {}}
    journal = [{"cursor": 0, "ts_ms": 10, "epoch": 1, "summary": "d0", **_e}]
    session = await mgr.warm_up_session("arm0", None, journal)
    assert session.cursor == 0

    # Fresh client sends state but empty journal — agent session should be reset
    state = {
        "ts_ms": 200,
        "epoch": 2,
        "cursor": -1,
        "active_target_handle": "bowl",
        "held_object_handle": "cube",
        "known_scene_handles": ["bowl", "cube", "bin"],
        "last_scene_description": "table with bowl, cube, and bin",
        "pending_operator_instruction": "Place the cube in the bin after picking the bowl",
        "recent_decisions": ["Pick the bowl first", "Then place the cube in the bin"],
        "last_snapshot_id": None,
        "last_arm_id": "arm0",
        "last_skill_phase": "VISUAL_ALIGN",
        "last_skill_name": "pick",
        "last_outcome_state": "in_progress",
        "recent_event_summaries": ["TARGET_ACQUIRED bowl", "SKILL_STARTED pick"],
        "goal_summary": "Pick bowl, then place cube in bin",
    }
    session = await mgr.warm_up_session("arm0", state, [])
    session.agent.reset_session.assert_awaited()
    assert session.cursor == -1
    assert len(session.context_store) == 0
    assert session.context_store._active_target_handle == "bowl"
    assert session.context_store._held_object_handle == "cube"
    assert session.context_store._known_scene_handles == ["bowl", "cube", "bin"]
    assert session.context_store._last_scene_description == "table with bowl, cube, and bin"
    assert session.context_store._pending_operator_instruction == "Place the cube in the bin after picking the bowl"
    assert session.pending_handoff is not None
    assert "Pick the bowl first" in session.pending_handoff
    assert "TARGET_ACQUIRED bowl" in session.pending_handoff
    assert "Goal summary: Pick bowl, then place cube in bin" in session.pending_handoff


@pytest.mark.asyncio
async def test_warm_up_incremental_preserves_earlier_entries(mgr: SessionManager):
    """Multiple incremental warm-up batches accumulate entries, not replace them."""
    _e = {"backend": "local", "entry_type": "decision", "data": {}}
    batch1 = [
        {"cursor": 0, "ts_ms": 10, "epoch": 1, "summary": "d0", **_e},
        {"cursor": 1, "ts_ms": 20, "epoch": 1, "summary": "d1", **_e},
    ]
    session = await mgr.warm_up_session("arm0", None, batch1)
    assert session.cursor == 1
    assert len(session.context_store) == 2

    batch2 = [
        {"cursor": 2, "ts_ms": 30, "epoch": 1, "summary": "d2", **_e},
        {"cursor": 3, "ts_ms": 40, "epoch": 1, "summary": "d3", **_e},
    ]
    session = await mgr.warm_up_session("arm0", None, batch2)
    assert session.cursor == 3
    # All 4 entries should be present, not just the last batch
    assert len(session.context_store) == 4


@pytest.mark.asyncio
async def test_warm_up_empty_journal_with_state_resets_store(mgr: SessionManager):
    """Full warm-up (state + empty journal) from a fresh client clears stale context."""
    _e = {"backend": "local", "entry_type": "decision", "data": {}}
    journal = [{"cursor": 0, "ts_ms": 10, "epoch": 1, "summary": "d0", **_e}]
    session = await mgr.warm_up_session("arm0", None, journal)
    assert session.cursor == 0
    assert len(session.context_store) == 1

    # Fresh client sends state but empty journal — stale context should be cleared
    state = {
        "ts_ms": 200,
        "epoch": 2,
        "cursor": -1,
        "active_target_handle": "cube",
        "held_object_handle": None,
        "known_scene_handles": ["cube", "bin"],
        "last_scene_description": "cube on the table, bin on the right",
        "pending_operator_instruction": "Pick the cube",
        "recent_decisions": ["Retry pick on cube"],
        "last_snapshot_id": None,
        "last_arm_id": "arm0",
        "last_skill_phase": "PLAN_APPROACH",
        "last_skill_name": "pick",
        "last_outcome_state": "in_progress",
        "recent_event_summaries": ["PERCEPTION_RECOVERED", "TARGET_ACQUIRED cube"],
        "goal_summary": "Pick the cube cleanly",
    }
    session = await mgr.warm_up_session("arm0", state, [])
    assert session.cursor == -1
    assert len(session.context_store) == 0
    assert session.context_store._known_scene_handles == ["cube", "bin"]
    assert session.pending_handoff is not None
    assert "Retry pick on cube" in session.pending_handoff
    assert "PERCEPTION_RECOVERED" in session.pending_handoff


@pytest.mark.asyncio
async def test_warm_up_state_only_preserves_rehydrated_history(prompts_dir: Path, mock_firestore_store: MagicMock):
    """State-only warm-up after Firestore rehydration preserves agent conversation history."""
    # Simulate a Firestore doc with msg_history (as if rehydrated from another instance)

    msg_recs = [
        {"msg_id": "a1", "role": "user", "text": "Pick the red cube", "ts_ms": 100, "is_summary": False},
        {"msg_id": "a2", "role": "model", "text": "Starting pick", "ts_ms": 200, "is_summary": False},
    ]
    doc = _make_firestore_doc(
        readiness="ready",
        cursor=0,
        msg_history=msg_recs,
        entries=[
            {
                "cursor": 0,
                "ts_ms": 10,
                "epoch": 1,
                "backend": "local",
                "entry_type": "decision",
                "summary": "d0",
                "data": {},
            },
        ],
    )
    mock_firestore_store.load = AsyncMock(return_value=doc)

    with patch("cloud_service.session_manager.PlannerAgent") as mock_cls:
        agent = _mock_agent()

        # After inject_compaction_state, agent should have records
        async def _fake_inject(summary, retained):
            for rec in retained:
                agent.msg_history.append(rec.role, rec.text)

        agent.inject_compaction_state = AsyncMock(side_effect=_fake_inject)
        mock_cls.return_value = agent

        mgr = SessionManager(
            model_name="test-model",
            prompts_dir=prompts_dir,
            vlm_fn_factory=lambda: MagicMock(),
            firestore_store=mock_firestore_store,
        )

        # get_or_create triggers rehydration (cache miss → Firestore load)
        await mgr.get_or_create("arm0")
        assert agent.msg_history.count() == 2  # rehydrated

        # Now warm-up with state but empty journal (e.g. reconnect after instance restart)
        state = {
            "ts_ms": 300,
            "epoch": 2,
            "cursor": -1,
            "active_target_handle": "cube",
            "held_object_handle": None,
            "known_scene_handles": ["cube"],
            "last_scene_description": "table",
            "pending_operator_instruction": None,
            "recent_decisions": [],
            "last_snapshot_id": None,
            "last_arm_id": "arm0",
            "last_skill_phase": None,
            "last_skill_name": None,
            "last_outcome_state": None,
        }
        await mgr.warm_up_session("arm0", state, [])

        # Agent history must NOT have been reset
        agent.reset_session.assert_not_awaited()
        assert agent.msg_history.count() == 2
        assert mgr.get_session("arm0").pending_handoff is None


@pytest.mark.asyncio
async def test_warm_up_empty_journal_no_state_preserves_store(mgr: SessionManager):
    """Incremental warm-up with empty batch (state=None) preserves existing context."""
    _e = {"backend": "local", "entry_type": "decision", "data": {}}
    journal = [{"cursor": 0, "ts_ms": 10, "epoch": 1, "summary": "d0", **_e}]
    session = await mgr.warm_up_session("arm0", None, journal)
    assert session.cursor == 0
    assert len(session.context_store) == 1

    # Incremental catch-up with no new entries — should be a no-op on context
    session = await mgr.warm_up_session("arm0", None, [])
    assert session.cursor == 0
    assert len(session.context_store) == 1


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
