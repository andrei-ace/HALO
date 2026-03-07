"""Tests for Mermaid stateDiagram-v2 parser."""

import pytest

from halo.contracts.enums import PhaseId, SkillName
from halo.services.skill_runner_service.mermaid_parser import parse_mermaid_fsm


def test_parse_pick_default():
    mmd = """\
stateDiagram-v2
    [*] --> SELECT_GRASP
    SELECT_GRASP --> PLAN_APPROACH : tracking_ok
    SELECT_GRASP --> DONE : timeout
    PLAN_APPROACH --> MOVE_PREGRASP : pass_through
    MOVE_PREGRASP --> VISUAL_ALIGN : close_enough
    MOVE_PREGRASP --> RECOVER_RETRY_APPROACH : target_lost
    MOVE_PREGRASP --> DONE : timeout
    VISUAL_ALIGN --> EXECUTE_APPROACH : close_enough
    VISUAL_ALIGN --> RECOVER_RETRY_APPROACH : target_lost
    VISUAL_ALIGN --> DONE : timeout
    EXECUTE_APPROACH --> CLOSE_GRIPPER : grasp_qualified
    EXECUTE_APPROACH --> RECOVER_RETRY_APPROACH : target_lost
    EXECUTE_APPROACH --> DONE : timeout
    CLOSE_GRIPPER --> VERIFY_GRASP : timer_elapsed
    CLOSE_GRIPPER --> LIFT : skip_verify
    VERIFY_GRASP --> LIFT : timer_elapsed
    LIFT --> DONE : success
    RECOVER_RETRY_APPROACH --> MOVE_PREGRASP : retry
    RECOVER_RETRY_APPROACH --> DONE : max_retries
"""
    graph = parse_mermaid_fsm(mmd, SkillName.PICK)
    assert graph.entry_node == "SELECT_GRASP"
    assert "DONE" in graph.terminal_nodes
    assert len(graph.nodes) == 10  # 9 states + DONE
    assert graph.nodes["SELECT_GRASP"].phase_id == PhaseId.SELECT_GRASP
    assert "PLAN_APPROACH" in graph.nodes["SELECT_GRASP"].successors
    assert "DONE" in graph.nodes["SELECT_GRASP"].successors


def test_parse_track_default():
    mmd = """\
stateDiagram-v2
    [*] --> ACQUIRING
    ACQUIRING --> DONE : tracking_ok
    ACQUIRING --> DONE : timeout
"""
    graph = parse_mermaid_fsm(mmd, SkillName.TRACK)
    assert graph.entry_node == "ACQUIRING"
    assert "DONE" in graph.terminal_nodes
    assert len(graph.nodes) == 2
    assert graph.nodes["ACQUIRING"].successors == ("DONE",)  # deduped


def test_parse_comments_ignored():
    mmd = """\
stateDiagram-v2
    %% this is a comment
    [*] --> ACQUIRING
    ACQUIRING --> DONE : ok
"""
    graph = parse_mermaid_fsm(mmd, SkillName.TRACK)
    assert graph.entry_node == "ACQUIRING"


def test_parse_no_entry_raises():
    mmd = "stateDiagram-v2\n    A --> B\n"
    with pytest.raises(ValueError, match="No entry node"):
        parse_mermaid_fsm(mmd, SkillName.PICK, phase_map={"A": PhaseId.IDLE, "B": PhaseId.DONE})


def test_parse_unknown_phase_raises():
    mmd = """\
stateDiagram-v2
    [*] --> UNKNOWN_NODE
    UNKNOWN_NODE --> DONE
"""
    with pytest.raises(ValueError, match="Phase mapping errors"):
        parse_mermaid_fsm(mmd, SkillName.PICK)


def test_parse_unlabeled_transition():
    mmd = """\
stateDiagram-v2
    [*] --> IDLE
    IDLE --> DONE
"""
    graph = parse_mermaid_fsm(mmd, SkillName.PICK)
    assert "DONE" in graph.nodes["IDLE"].successors


def test_graph_validate_catches_orphans():
    """Graph validation catches unreachable nodes."""
    from halo.services.skill_runner_service.graph import FsmGraph, FsmNode

    graph = FsmGraph(
        skill_name=SkillName.PICK,
        variant="test",
        nodes={
            "A": FsmNode(name="A", phase_id=PhaseId.IDLE, successors=("B",)),
            "B": FsmNode(name="B", phase_id=PhaseId.DONE, successors=()),
            "C": FsmNode(name="C", phase_id=PhaseId.SELECT_GRASP, successors=()),  # orphan
        },
        edges=(),
        entry_node="A",
        terminal_nodes=frozenset({"B", "C"}),
        mermaid_source="",
    )
    errors = graph.validate()
    assert any("unreachable" in e and "'C'" in e for e in errors)


def test_parse_from_file():
    """Parse actual .mmd files from configs/skills/."""
    from pathlib import Path

    skills_dir = Path(__file__).resolve().parent.parent / "configs" / "skills"
    pick_graph = parse_mermaid_fsm((skills_dir / "pick" / "default.mmd").read_text(), SkillName.PICK)
    assert pick_graph.entry_node == "SELECT_GRASP"
    assert len(pick_graph.nodes) == 10

    track_graph = parse_mermaid_fsm((skills_dir / "track" / "default.mmd").read_text(), SkillName.TRACK)
    assert track_graph.entry_node == "ACQUIRING"
    assert len(track_graph.nodes) == 2
