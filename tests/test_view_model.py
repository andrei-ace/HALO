"""Tests for FsmViewModel."""

from halo.contracts.enums import SkillName
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.definitions import build_default_registry
from halo.services.skill_runner_service.engine import FsmEngine
from halo.services.skill_runner_service.queue import SkillQueue
from halo.services.skill_runner_service.skill_run import QueuedSkill
from halo.services.skill_runner_service.view_model import build_fsm_view_model

T0 = 1000
registry = build_default_registry()


def _make_run():
    defn = registry.get(SkillName.PICK)
    engine = FsmEngine(defn.graph, defn.handler_factory(), SkillRunnerConfig(), defn.global_guard_factory())
    return engine.create_run(T0, "run-1", "obj-1")


def test_view_model_none_when_no_active():
    q = SkillQueue()
    vm = build_fsm_view_model(None, q, T0)
    assert vm is None


def test_view_model_basic():
    run = _make_run()
    q = SkillQueue()
    vm = build_fsm_view_model(run, q, T0 + 100)
    assert vm is not None
    assert vm.skill_name == "PICK"
    assert vm.current_node == "SELECT_GRASP"
    assert vm.outcome == "IN_PROGRESS"
    assert vm.failure_code is None
    assert len(vm.nodes) == 11
    # Active node should have elapsed_ms
    active_nodes = [n for n in vm.nodes if n.name == "SELECT_GRASP"]
    assert active_nodes[0].elapsed_ms == 100


def test_view_model_with_queued():
    run = _make_run()
    q = SkillQueue()
    q.enqueue(QueuedSkill(SkillName.TRACK, "run-2", "obj-2", "default", {}, T0))
    q.enqueue(QueuedSkill(SkillName.PICK, "run-3", "obj-3", "default", {}, T0))
    vm = build_fsm_view_model(run, q, T0)
    assert len(vm.queued_skills) == 2
    assert vm.queued_skills[0].position == 0
    assert vm.queued_skills[1].position == 1


def test_view_model_mermaid_source():
    run = _make_run()
    q = SkillQueue()
    vm = build_fsm_view_model(run, q, T0)
    assert "stateDiagram-v2" in vm.mermaid_source


def test_view_model_edges():
    run = _make_run()
    q = SkillQueue()
    vm = build_fsm_view_model(run, q, T0)
    assert len(vm.edges) > 0
    assert any(e.source == "SELECT_GRASP" for e in vm.edges)
