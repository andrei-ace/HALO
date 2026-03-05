"""Unit tests for SessionManager — per-arm session management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cloud_service.session_manager import ArmSession, SessionManager


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create minimal prompts directory."""
    d = tmp_path / "planner"
    d.mkdir()
    (d / "system_prompt.md").write_text("You are a planner.")
    return d


@pytest.fixture
def mgr(prompts_dir: Path) -> SessionManager:
    """SessionManager with mocked PlannerAgent creation."""
    with patch("cloud_service.session_manager.PlannerAgent") as mock_cls:
        mock_cls.return_value = MagicMock(decide=MagicMock(), last_reasoning="", reset_loop_state=MagicMock())
        yield SessionManager(
            model_name="test-model",
            prompts_dir=prompts_dir,
            vlm_fn_factory=lambda: MagicMock(),
            max_sessions=4,
            idle_timeout_s=0.001,  # very short for testing
        )


def test_get_or_create_new_session(mgr: SessionManager):
    session = mgr.get_or_create("arm0")
    assert isinstance(session, ArmSession)
    assert session.arm_id == "arm0"
    assert mgr.session_count == 1


def test_get_or_create_returns_existing(mgr: SessionManager):
    s1 = mgr.get_or_create("arm0")
    s2 = mgr.get_or_create("arm0")
    assert s1 is s2
    assert mgr.session_count == 1


def test_multiple_sessions(mgr: SessionManager):
    mgr.get_or_create("arm0")
    mgr.get_or_create("arm1")
    mgr.get_or_create("arm2")
    assert mgr.session_count == 3


def test_eviction_at_capacity(prompts_dir: Path):
    """When at max_sessions, LRU session is evicted."""
    with patch("cloud_service.session_manager.PlannerAgent") as mock_cls:
        mock_cls.return_value = MagicMock(decide=MagicMock(), last_reasoning="", reset_loop_state=MagicMock())
        # Use a long idle timeout so idle eviction doesn't interfere
        mgr = SessionManager(
            model_name="test-model",
            prompts_dir=prompts_dir,
            vlm_fn_factory=lambda: MagicMock(),
            max_sessions=4,
            idle_timeout_s=600.0,
        )
        mgr.get_or_create("arm0")
        mgr.get_or_create("arm1")
        mgr.get_or_create("arm2")
        mgr.get_or_create("arm3")
        assert mgr.session_count == 4
        # At capacity (4). Creating arm4 should evict arm0 (LRU)
        mgr.get_or_create("arm4")
        assert mgr.session_count == 4
        assert mgr.get_session("arm0") is None
        assert mgr.get_session("arm4") is not None


def test_warm_up_session(mgr: SessionManager):
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

    session = mgr.warm_up_session("arm0", state_dict, journal)
    assert session.readiness == "ready"
    assert session.cursor == 1
    assert session.context_store._active_target_handle == "cube"


def test_warm_up_incremental(mgr: SessionManager):
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
    session = mgr.warm_up_session("arm0", None, journal1)
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
    session = mgr.warm_up_session("arm0", None, journal2)
    assert session.cursor == 2


def test_reset_session(mgr: SessionManager):
    mgr.get_or_create("arm0")
    mgr.reset_session("arm0")
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
