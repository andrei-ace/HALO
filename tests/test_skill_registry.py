"""Tests for SkillRegistry and build_default_registry."""

from halo.contracts.enums import PhaseId, SkillName
from halo.services.skill_runner_service.definitions import build_default_registry


def test_default_registry_has_pick_and_track():
    r = build_default_registry()
    assert r.get(SkillName.PICK, "default") is not None
    assert r.get(SkillName.TRACK, "default") is not None


def test_default_registry_pick_graph():
    r = build_default_registry()
    defn = r.get(SkillName.PICK)
    assert defn.graph.entry_node == "SELECT_GRASP"
    assert "DONE" in defn.graph.terminal_nodes
    assert defn.graph.nodes["SELECT_GRASP"].phase_id == PhaseId.SELECT_GRASP


def test_default_registry_track_graph():
    r = build_default_registry()
    defn = r.get(SkillName.TRACK)
    assert defn.graph.entry_node == "ACQUIRING"
    assert "DONE" in defn.graph.terminal_nodes


def test_list_variants():
    r = build_default_registry()
    pick_variants = r.list_variants(SkillName.PICK)
    assert "default" in pick_variants
    track_variants = r.list_variants(SkillName.TRACK)
    assert "default" in track_variants


def test_unknown_skill_returns_none():
    r = build_default_registry()
    assert r.get(SkillName.PLACE) is None


def test_unknown_variant_returns_none():
    r = build_default_registry()
    assert r.get(SkillName.PICK, "custom") is None


def test_handler_factories_return_handlers():
    r = build_default_registry()
    defn = r.get(SkillName.PICK)
    handlers = defn.handler_factory()
    assert "SELECT_GRASP" in handlers
    assert "LIFT" in handlers

    defn = r.get(SkillName.TRACK)
    handlers = defn.handler_factory()
    assert "ACQUIRING" in handlers


def test_global_guard_factories():
    r = build_default_registry()
    defn = r.get(SkillName.PICK)
    guards = defn.global_guard_factory()
    assert len(guards) >= 1  # ReacquireFailedGuard

    defn = r.get(SkillName.TRACK)
    guards = defn.global_guard_factory()
    assert len(guards) == 0  # TRACK has no global guards
