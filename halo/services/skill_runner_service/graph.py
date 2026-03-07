from __future__ import annotations

from dataclasses import dataclass

from halo.contracts.enums import PhaseId, SkillName


@dataclass(frozen=True)
class FsmEdge:
    source: str
    target: str
    label: str


@dataclass(frozen=True)
class FsmNode:
    name: str
    phase_id: PhaseId
    successors: tuple[str, ...]


@dataclass(frozen=True)
class FsmGraph:
    skill_name: SkillName
    variant: str
    nodes: dict[str, FsmNode]
    edges: tuple[FsmEdge, ...]
    entry_node: str
    terminal_nodes: frozenset[str]
    mermaid_source: str

    def validate(self) -> list[str]:
        errors: list[str] = []

        # Entry node must exist
        if self.entry_node not in self.nodes:
            errors.append(f"entry_node '{self.entry_node}' not in nodes")

        # Terminal nodes must exist
        for tn in self.terminal_nodes:
            if tn not in self.nodes:
                errors.append(f"terminal_node '{tn}' not in nodes")

        # All edge targets/sources must exist
        for edge in self.edges:
            if edge.source not in self.nodes:
                errors.append(f"edge source '{edge.source}' not in nodes")
            if edge.target not in self.nodes:
                errors.append(f"edge target '{edge.target}' not in nodes")

        # All successor references must exist
        for node in self.nodes.values():
            for succ in node.successors:
                if succ not in self.nodes:
                    errors.append(f"node '{node.name}' successor '{succ}' not in nodes")

        # Check reachability from entry_node
        if self.entry_node in self.nodes:
            reachable: set[str] = set()
            stack = [self.entry_node]
            while stack:
                n = stack.pop()
                if n in reachable:
                    continue
                reachable.add(n)
                if n in self.nodes:
                    for succ in self.nodes[n].successors:
                        stack.append(succ)
            orphans = set(self.nodes.keys()) - reachable
            for orph in sorted(orphans):
                errors.append(f"node '{orph}' is unreachable from entry_node '{self.entry_node}'")

        return errors
