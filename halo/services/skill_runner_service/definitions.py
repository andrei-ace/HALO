from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from halo.contracts.enums import SkillName
from halo.services.skill_runner_service.graph import FsmGraph
from halo.services.skill_runner_service.handlers import (
    GlobalGuard,
    StateHandler,
    build_pick_global_guards,
    build_pick_handlers,
    build_place_global_guards,
    build_place_handlers,
    build_track_global_guards,
    build_track_handlers,
)
from halo.services.skill_runner_service.mermaid_parser import parse_mermaid_fsm

_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "skills"


@dataclass(frozen=True)
class SkillDefinition:
    skill_name: SkillName
    variant: str
    graph: FsmGraph
    handler_factory: Callable[[], dict[str, StateHandler]]
    global_guard_factory: Callable[[], list[GlobalGuard]]


@dataclass
class SkillRegistry:
    _definitions: dict[tuple[SkillName, str], SkillDefinition] = field(default_factory=dict)

    def register(self, defn: SkillDefinition) -> None:
        self._definitions[(defn.skill_name, defn.variant)] = defn

    def get(self, skill_name: SkillName, variant: str = "default") -> SkillDefinition | None:
        return self._definitions.get((skill_name, variant))

    def list_variants(self, skill_name: SkillName) -> list[str]:
        return [v for (sn, v) in self._definitions if sn == skill_name]


def build_default_registry() -> SkillRegistry:
    registry = SkillRegistry()

    # PICK:default
    pick_mmd = (_SKILLS_DIR / "pick" / "default.mmd").read_text()
    pick_graph = parse_mermaid_fsm(pick_mmd, SkillName.PICK, variant="default")
    registry.register(
        SkillDefinition(
            skill_name=SkillName.PICK,
            variant="default",
            graph=pick_graph,
            handler_factory=build_pick_handlers,
            global_guard_factory=build_pick_global_guards,
        )
    )

    # TRACK:default
    track_mmd = (_SKILLS_DIR / "track" / "default.mmd").read_text()
    track_graph = parse_mermaid_fsm(track_mmd, SkillName.TRACK, variant="default")
    registry.register(
        SkillDefinition(
            skill_name=SkillName.TRACK,
            variant="default",
            graph=track_graph,
            handler_factory=build_track_handlers,
            global_guard_factory=build_track_global_guards,
        )
    )

    # PLACE:default
    place_mmd = (_SKILLS_DIR / "place" / "default.mmd").read_text()
    place_graph = parse_mermaid_fsm(place_mmd, SkillName.PLACE, variant="default")
    registry.register(
        SkillDefinition(
            skill_name=SkillName.PLACE,
            variant="default",
            graph=place_graph,
            handler_factory=build_place_handlers,
            global_guard_factory=build_place_global_guards,
        )
    )

    return registry
