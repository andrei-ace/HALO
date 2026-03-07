"""Tests for FsmGraph validation."""

from halo.contracts.enums import PhaseId, SkillName
from halo.services.skill_runner_service.graph import FsmEdge, FsmGraph, FsmNode


def _make_graph(**kwargs) -> FsmGraph:
    defaults = dict(
        skill_name=SkillName.PICK,
        variant="test",
        nodes={
            "A": FsmNode(name="A", phase_id=PhaseId.IDLE, successors=("B",)),
            "B": FsmNode(name="B", phase_id=PhaseId.DONE, successors=()),
        },
        edges=(FsmEdge(source="A", target="B", label="go"),),
        entry_node="A",
        terminal_nodes=frozenset({"B"}),
        mermaid_source="",
    )
    defaults.update(kwargs)
    return FsmGraph(**defaults)


def test_valid_graph_has_no_errors():
    assert _make_graph().validate() == []


def test_missing_entry_node():
    g = _make_graph(entry_node="MISSING")
    errors = g.validate()
    assert any("entry_node" in e for e in errors)


def test_missing_terminal_node():
    g = _make_graph(terminal_nodes=frozenset({"MISSING"}))
    errors = g.validate()
    assert any("terminal_node" in e for e in errors)


def test_edge_with_missing_source():
    g = _make_graph(edges=(FsmEdge(source="MISSING", target="B", label="x"),))
    errors = g.validate()
    assert any("edge source" in e for e in errors)


def test_edge_with_missing_target():
    g = _make_graph(edges=(FsmEdge(source="A", target="MISSING", label="x"),))
    errors = g.validate()
    assert any("edge target" in e for e in errors)


def test_successor_with_missing_node():
    g = _make_graph(
        nodes={
            "A": FsmNode(name="A", phase_id=PhaseId.IDLE, successors=("MISSING",)),
            "B": FsmNode(name="B", phase_id=PhaseId.DONE, successors=()),
        }
    )
    errors = g.validate()
    assert any("successor 'MISSING'" in e for e in errors)
