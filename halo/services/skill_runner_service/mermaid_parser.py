from __future__ import annotations

import re

from halo.contracts.enums import PhaseId, SkillName
from halo.services.skill_runner_service.graph import FsmEdge, FsmGraph, FsmNode

_TRANSITION_RE = re.compile(
    r"^\s*"
    r"(?P<source>\[\*\]|[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*-->\s*"
    r"(?P<target>\[\*\]|[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s*:\s*(?P<label>.+?))?"
    r"\s*$"
)

# Default node name -> PhaseId mapping
_DEFAULT_PHASE_MAP: dict[str, PhaseId] = {member.name: member for member in PhaseId}


def parse_mermaid_fsm(
    mermaid_text: str,
    skill_name: SkillName,
    variant: str = "default",
    phase_map: dict[str, PhaseId] | None = None,
) -> FsmGraph:
    if phase_map is None:
        phase_map = _DEFAULT_PHASE_MAP

    entry_node: str | None = None
    terminal_targets: set[str] = set()
    edges: list[FsmEdge] = []
    node_names: set[str] = set()
    successors_map: dict[str, list[str]] = {}

    for line in mermaid_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%") or stripped == "stateDiagram-v2":
            continue

        m = _TRANSITION_RE.match(stripped)
        if not m:
            continue

        source = m.group("source")
        target = m.group("target")
        label = (m.group("label") or "").strip()

        # [*] --> NODE  →  entry
        if source == "[*]":
            if entry_node is not None and entry_node != target:
                raise ValueError(f"Multiple entry nodes: '{entry_node}' and '{target}'")
            entry_node = target
            node_names.add(target)
            continue

        # NODE --> [*]  →  terminal (maps to DONE)
        if target == "[*]":
            terminal_targets.add(source)
            node_names.add(source)
            # Map [*] terminal to DONE
            successors_map.setdefault(source, []).append("DONE")
            edges.append(FsmEdge(source=source, target="DONE", label=label))
            node_names.add("DONE")
            continue

        # Normal transition
        node_names.add(source)
        node_names.add(target)
        successors_map.setdefault(source, []).append(target)
        edges.append(FsmEdge(source=source, target=target, label=label))

    if entry_node is None:
        raise ValueError("No entry node found (missing [*] --> NODE)")

    # Validate all nodes map to PhaseId
    errors: list[str] = []
    for name in sorted(node_names):
        if name not in phase_map:
            errors.append(f"node '{name}' has no PhaseId mapping")
    if errors:
        raise ValueError("Phase mapping errors: " + "; ".join(errors))

    # Build nodes
    nodes: dict[str, FsmNode] = {}
    for name in node_names:
        succs = tuple(dict.fromkeys(successors_map.get(name, [])))  # dedup preserving order
        nodes[name] = FsmNode(name=name, phase_id=phase_map[name], successors=succs)

    # Terminal nodes: nodes with no successors or explicit DONE
    terminal_nodes = frozenset(name for name, node in nodes.items() if not node.successors)

    graph = FsmGraph(
        skill_name=skill_name,
        variant=variant,
        nodes=nodes,
        edges=tuple(edges),
        entry_node=entry_node,
        terminal_nodes=terminal_nodes,
        mermaid_source=mermaid_text,
    )

    validation_errors = graph.validate()
    if validation_errors:
        raise ValueError("Graph validation failed: " + "; ".join(validation_errors))

    return graph
