"""Unit tests for ContextStore."""

from __future__ import annotations

from halo.cognitive.context_store import ContextEntry, ContextSnapshot, ContextStore


def test_append_and_len():
    store = ContextStore()
    store.append(epoch=1, backend="local", entry_type="decision", summary="Issued start_skill pick")
    store.append(
        epoch=1,
        backend="local",
        entry_type="scene",
        summary="Table with red cube",
        data={"handles": ["red_cube_01"]},
    )
    assert len(store) == 2


def test_cursor_monotonic():
    store = ContextStore()
    e1 = store.append(epoch=1, backend="local", entry_type="decision", summary="d1")
    e2 = store.append(epoch=1, backend="local", entry_type="decision", summary="d2")
    assert e1.cursor == 0
    assert e2.cursor == 1
    assert store.latest_cursor == 1


def test_trim_at_max_entries():
    store = ContextStore(max_entries=5)
    for i in range(10):
        store.append(epoch=1, backend="local", entry_type="decision", summary=f"d{i}")
    assert len(store) == 5
    # Oldest entries should have been trimmed
    assert store._entries[0].summary == "d5"


def test_take_snapshot():
    store = ContextStore()
    store.append(
        epoch=1,
        backend="local",
        entry_type="scene",
        summary="A table with objects",
        data={"handles": ["red_cube_01", "green_cube_01"]},
    )
    store.append(epoch=1, backend="local", entry_type="decision", summary="Issued describe_scene")
    store.set_active_target("red_cube_01")
    store.set_held_object(None)

    snap = store.take_snapshot(epoch=1)
    assert isinstance(snap, ContextSnapshot)
    assert snap.epoch == 1
    assert snap.active_target_handle == "red_cube_01"
    assert snap.held_object_handle is None
    assert snap.known_scene_handles == ["red_cube_01", "green_cube_01"]
    assert snap.last_scene_description == "A table with objects"
    assert "Issued describe_scene" in snap.recent_decisions


def test_take_snapshot_recent_decisions_limited():
    store = ContextStore()
    for i in range(10):
        store.append(epoch=1, backend="local", entry_type="decision", summary=f"decision_{i}")

    snap = store.take_snapshot(epoch=1)
    assert len(snap.recent_decisions) == 5
    assert snap.recent_decisions[0] == "decision_5"
    assert snap.recent_decisions[-1] == "decision_9"


def test_scene_entry_updates_state():
    store = ContextStore()
    store.append(epoch=1, backend="local", entry_type="scene", summary="Table with cube", data={"handles": ["cube_01"]})
    assert store._known_scene_handles == ["cube_01"]
    assert store._last_scene_description == "Table with cube"


def test_operator_entry_sets_pending():
    store = ContextStore()
    store.append(epoch=1, backend="local", entry_type="operator", summary="Pick the red cube")
    assert store._pending_operator_instruction == "Pick the red cube"


def test_decision_clears_pending_operator():
    store = ContextStore()
    store.append(epoch=1, backend="local", entry_type="operator", summary="Pick the red cube")
    assert store._pending_operator_instruction == "Pick the red cube"
    store.append(epoch=1, backend="local", entry_type="decision", summary="Issued start_skill")
    assert store._pending_operator_instruction is None


def test_get_handoff_context():
    store = ContextStore()
    store.append(
        epoch=1,
        backend="local",
        entry_type="scene",
        summary="Table with red cube and green cube",
        data={"handles": ["red_cube_01", "green_cube_01"]},
    )
    store.append(epoch=1, backend="local", entry_type="decision", summary="Issued track_object red_cube_01")
    store.append(epoch=1, backend="local", entry_type="decision", summary="Issued start_skill pick red_cube_01")
    store.set_active_target("red_cube_01")
    store.set_held_object("red_cube_01")

    ctx = store.get_handoff_context(epoch=2)
    assert "[Context handoff from previous backend]" in ctx
    assert "red_cube_01" in ctx
    assert "green_cube_01" in ctx
    assert "Currently holding: red_cube_01" in ctx
    assert "start_skill pick" in ctx


def test_get_handoff_context_empty():
    store = ContextStore()
    ctx = store.get_handoff_context(epoch=1)
    assert "[Context handoff from previous backend]" in ctx


def test_get_entries_after():
    store = ContextStore()
    store.append(epoch=1, backend="local", entry_type="decision", summary="d0")
    store.append(epoch=1, backend="local", entry_type="decision", summary="d1")
    store.append(epoch=1, backend="local", entry_type="decision", summary="d2")

    entries = store.get_entries_after(cursor=0)
    assert len(entries) == 2
    assert entries[0].summary == "d1"
    assert entries[1].summary == "d2"


def test_get_entries_after_with_limit():
    store = ContextStore()
    for i in range(10):
        store.append(epoch=1, backend="local", entry_type="decision", summary=f"d{i}")

    entries = store.get_entries_after(cursor=5, limit=2)
    assert len(entries) == 2
    assert entries[0].summary == "d6"


def test_get_entries_after_minus_one():
    """cursor=-1 should return all entries."""
    store = ContextStore()
    store.append(epoch=1, backend="local", entry_type="decision", summary="d0")
    store.append(epoch=1, backend="local", entry_type="decision", summary="d1")

    entries = store.get_entries_after(cursor=-1)
    assert len(entries) == 2


def test_context_entry_frozen():
    entry = ContextEntry(cursor=0, ts_ms=1000, epoch=1, backend="local", entry_type="decision", summary="test")
    assert entry.cursor == 0
    # Frozen dataclass — should not be mutable
    try:
        entry.cursor = 5  # type: ignore[misc]
        assert False, "Expected FrozenInstanceError"
    except AttributeError:
        pass
