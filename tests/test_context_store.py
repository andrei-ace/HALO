"""Unit tests for ContextStore."""

from __future__ import annotations

from halo.cognitive.context_store import CognitiveState, ContextEntry, ContextSnapshot, ContextStore


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


# ---------------------------------------------------------------------------
# CognitiveState
# ---------------------------------------------------------------------------


def test_build_cognitive_state_no_snapshot():
    store = ContextStore()
    store.append(epoch=1, backend="local", entry_type="scene", summary="Table", data={"handles": ["cube_01"]})
    store.append(epoch=1, backend="local", entry_type="decision", summary="Issued describe_scene")
    store.set_active_target("cube_01")

    state = store.build_cognitive_state(epoch=1)
    assert isinstance(state, CognitiveState)
    assert state.epoch == 1
    assert state.cursor == 1
    assert state.active_target_handle == "cube_01"
    assert state.known_scene_handles == ["cube_01"]
    assert state.last_scene_description == "Table"
    assert "Issued describe_scene" in state.recent_decisions
    # No snapshot → runtime fields are None
    assert state.last_snapshot_id is None
    assert state.last_arm_id is None
    assert state.last_skill_phase is None
    assert state.last_skill_name is None
    assert state.last_outcome_state is None


def test_build_cognitive_state_with_snapshot():
    from halo.contracts.enums import (
        ActStatus,
        PerceptionFailureCode,
        SafetyState,
        SkillName,
        SkillOutcomeState,
        TrackingStatus,
    )
    from halo.contracts.snapshots import (
        ActInfo,
        OutcomeInfo,
        PerceptionInfo,
        PlannerSnapshot,
        ProgressInfo,
        SafetyInfo,
        SkillInfo,
        TargetInfo,
    )

    store = ContextStore()
    store.append(epoch=2, backend="cloud", entry_type="decision", summary="Started pick")

    from halo.contracts.enums import PhaseId

    snap = PlannerSnapshot(
        snapshot_id="snap-42",
        ts_ms=1000,
        arm_id="arm0",
        skill=SkillInfo(name=SkillName.PICK, skill_run_id="run-1", phase=PhaseId.EXECUTE_APPROACH),
        target=TargetInfo(
            handle="cube_01",
            hint_valid=True,
            confidence=0.9,
            obs_age_ms=10,
            time_skew_ms=2,
            delta_xyz_ee=(0.0, 0.0, 0.0),
            distance_m=0.1,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=200, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=5000, no_progress_ms=0, delta_distance=-0.02),
        outcome=OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False),
        safety=SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=()),
        command_acks=(),
        recent_events=(),
    )

    state = store.build_cognitive_state(epoch=2, snapshot=snap)
    assert state.last_snapshot_id == "snap-42"
    assert state.last_arm_id == "arm0"
    assert state.last_skill_phase == "EXECUTE_APPROACH"
    assert state.last_skill_name == "PICK"
    assert state.last_outcome_state == "IN_PROGRESS"


def test_build_cognitive_state_empty_store():
    store = ContextStore()
    state = store.build_cognitive_state(epoch=0)
    assert state.cursor == -1
    assert state.recent_decisions == []
    assert state.active_target_handle is None


# ---------------------------------------------------------------------------
# apply_entries
# ---------------------------------------------------------------------------


def test_apply_entries_basic():
    store = ContextStore()
    entries = [
        ContextEntry(cursor=0, ts_ms=100, epoch=1, backend="local", entry_type="decision", summary="d0"),
        ContextEntry(
            cursor=1,
            ts_ms=200,
            epoch=1,
            backend="local",
            entry_type="scene",
            summary="Table",
            data={"handles": ["cube"]},
        ),
    ]
    store.apply_entries(entries)
    assert len(store) == 2
    assert store.latest_cursor == 1
    assert store._known_scene_handles == ["cube"]
    assert store._last_scene_description == "Table"


def test_apply_entries_updates_tracked_state():
    store = ContextStore()
    entries = [
        ContextEntry(
            cursor=0,
            ts_ms=100,
            epoch=1,
            backend="local",
            entry_type="operator",
            summary="Pick the cube",
        ),
        ContextEntry(
            cursor=1,
            ts_ms=200,
            epoch=1,
            backend="local",
            entry_type="decision",
            summary="Issued start_skill",
        ),
    ]
    store.apply_entries(entries)
    # operator sets pending, decision clears it
    assert store._pending_operator_instruction is None


def test_apply_entries_cursor_monotonicity_violation():
    store = ContextStore()
    store.append(epoch=1, backend="local", entry_type="decision", summary="d0")
    # cursor=0 already exists, trying to apply cursor=0 again should fail
    import pytest

    with pytest.raises(ValueError, match="monotonicity"):
        store.apply_entries(
            [
                ContextEntry(cursor=0, ts_ms=200, epoch=1, backend="local", entry_type="decision", summary="dup"),
            ]
        )


def test_apply_entries_incremental():
    """Apply entries after some already exist."""
    store = ContextStore()
    store.append(epoch=1, backend="local", entry_type="decision", summary="d0")
    store.append(epoch=1, backend="local", entry_type="decision", summary="d1")
    assert store.latest_cursor == 1

    # Now apply entries with cursor > 1
    entries = [
        ContextEntry(cursor=2, ts_ms=300, epoch=1, backend="cloud", entry_type="decision", summary="d2"),
        ContextEntry(cursor=3, ts_ms=400, epoch=1, backend="cloud", entry_type="decision", summary="d3"),
    ]
    store.apply_entries(entries)
    assert len(store) == 4
    assert store.latest_cursor == 3


def test_apply_entries_trims():
    store = ContextStore(max_entries=3)
    entries = [
        ContextEntry(cursor=i, ts_ms=i * 100, epoch=1, backend="local", entry_type="decision", summary=f"d{i}")
        for i in range(5)
    ]
    store.apply_entries(entries)
    assert len(store) == 3
    assert store._entries[0].summary == "d2"
