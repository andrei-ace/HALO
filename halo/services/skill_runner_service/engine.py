from __future__ import annotations

from halo.contracts.enums import PhaseId, SkillFailureCode, SkillOutcomeState
from halo.contracts.snapshots import ActInfo, PerceptionInfo, TargetInfo
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.graph import FsmGraph
from halo.services.skill_runner_service.handlers import GlobalGuard, HandlerResult, StateContext, StateHandler
from halo.services.skill_runner_service.skill_run import NodeStatus, SkillRun, TransitionRecord


class FsmEngine:
    def __init__(
        self,
        graph: FsmGraph,
        handlers: dict[str, StateHandler],
        config: SkillRunnerConfig,
        global_guards: list[GlobalGuard] | None = None,
    ) -> None:
        self._graph = graph
        self._handlers = handlers
        self._config = config
        self._global_guards = global_guards or []

    def create_run(
        self,
        now_ms: int,
        skill_run_id: str,
        target_handle: str,
        variant: str = "default",
    ) -> SkillRun:
        entry = self._graph.entry_node
        node_statuses = {name: NodeStatus.PENDING for name in self._graph.nodes}
        node_statuses[entry] = NodeStatus.ACTIVE

        return SkillRun(
            skill_run_id=skill_run_id,
            skill_name=self._graph.skill_name,
            variant=variant,
            target_handle=target_handle,
            graph=self._graph,
            current_node=entry,
            node_statuses=node_statuses,
            state_bag={},
            phase_start_ms=now_ms,
            skill_start_ms=now_ms,
            outcome=SkillOutcomeState.IN_PROGRESS,
            failure_code=None,
            transition_history=[],
        )

    def advance(
        self,
        run: SkillRun,
        now_ms: int,
        target: TargetInfo | None,
        perception: PerceptionInfo,
        act: ActInfo,
        held_object_handle: str | None = None,
    ) -> PhaseId | None:
        """Returns old PhaseId on transition, None if stayed. One transition per tick max."""
        if not run.is_active:
            return None

        ctx = self._build_context(run, now_ms, target, perception, act, held_object_handle=held_object_handle)

        # Global guards
        for guard in self._global_guards:
            result = guard.check(ctx)
            if result is not None:
                return self._apply_result(run, now_ms, result)

        # Node handler
        handler = self._handlers.get(run.current_node)
        if handler is None:
            return None

        result = handler.evaluate(ctx)
        return self._apply_result(run, now_ms, result)

    def sync_phase(self, run: SkillRun, now_ms: int, phase_id: PhaseId) -> PhaseId | None:
        """Accept external phase. Returns old PhaseId on transition, None if no change."""
        if not run.is_active or run.phase_id == phase_id:
            return None
        if phase_id == PhaseId.DONE:
            return self._succeed(run, now_ms)
        # Ignore backward transitions
        if phase_id.value < run.phase_id.value:
            return None
        # Find node matching phase_id
        target_node = self._find_node_by_phase(phase_id)
        if target_node is None:
            return None
        return self._transition(run, now_ms, target_node, trigger="sync_phase")

    def abort(self, run: SkillRun, now_ms: int, code: SkillFailureCode = SkillFailureCode.UNSAFE_ABORT) -> None:
        if run.current_node in self._graph.terminal_nodes and not run.is_active:
            return
        old_phase = run.phase_id
        run.node_statuses[run.current_node] = NodeStatus.FAILED
        run.failure_code = code
        run.outcome = SkillOutcomeState.FAILURE
        # Move to DONE
        done_node = self._find_node_by_phase(PhaseId.DONE) or "DONE"
        run.transition_history.append(TransitionRecord(run.current_node, done_node, now_ms, "abort"))
        if done_node in run.node_statuses:
            run.node_statuses[done_node] = NodeStatus.ACTIVE
        run.current_node = done_node
        run.phase_start_ms = now_ms
        # old_phase recorded in transition_history
        _ = old_phase

    def fail(self, run: SkillRun, now_ms: int, code: SkillFailureCode) -> None:
        if run.current_node in self._graph.terminal_nodes and not run.is_active:
            return
        run.node_statuses[run.current_node] = NodeStatus.FAILED
        run.failure_code = code
        run.outcome = SkillOutcomeState.FAILURE
        done_node = self._find_node_by_phase(PhaseId.DONE) or "DONE"
        run.transition_history.append(TransitionRecord(run.current_node, done_node, now_ms, f"fail:{code.value}"))
        if done_node in run.node_statuses:
            run.node_statuses[done_node] = NodeStatus.ACTIVE
        run.current_node = done_node
        run.phase_start_ms = now_ms

    def needs_chunk(self, run: SkillRun, act: ActInfo) -> bool:
        return act.buffer_fill_ms < self._config.buffer_target_ms

    # --- Private ---

    def _build_context(
        self,
        run: SkillRun,
        now_ms: int,
        target: TargetInfo | None,
        perception: PerceptionInfo,
        act: ActInfo,
        held_object_handle: str | None = None,
    ) -> StateContext:
        node = self._graph.nodes[run.current_node]
        return StateContext(
            now_ms=now_ms,
            elapsed_ms=now_ms - run.phase_start_ms,
            target=target,
            perception=perception,
            act=act,
            config=self._config,
            state_bag=run.state_bag,
            target_handle=run.target_handle,
            successors=node.successors,
            held_object_handle=held_object_handle,
        )

    def _apply_result(self, run: SkillRun, now_ms: int, result: HandlerResult) -> PhaseId | None:
        if result.fail_code is not None:
            old = run.phase_id
            self.fail(run, now_ms, result.fail_code)
            return old
        if result.succeed:
            return self._succeed(run, now_ms)
        if result.transition_to is not None:
            return self._transition(run, now_ms, result.transition_to, trigger=result.trigger)
        return None

    def _transition(self, run: SkillRun, now_ms: int, target_node: str, trigger: str = "") -> PhaseId:
        old_phase = run.phase_id
        run.node_statuses[run.current_node] = NodeStatus.COMPLETED
        run.transition_history.append(TransitionRecord(run.current_node, target_node, now_ms, trigger))
        run.current_node = target_node
        run.node_statuses[target_node] = NodeStatus.ACTIVE
        run.phase_start_ms = now_ms
        return old_phase

    def _succeed(self, run: SkillRun, now_ms: int) -> PhaseId:
        old_phase = run.phase_id
        run.node_statuses[run.current_node] = NodeStatus.COMPLETED
        run.outcome = SkillOutcomeState.SUCCESS
        done_node = self._find_node_by_phase(PhaseId.DONE) or "DONE"
        run.transition_history.append(TransitionRecord(run.current_node, done_node, now_ms, "success"))
        if done_node in run.node_statuses:
            run.node_statuses[done_node] = NodeStatus.ACTIVE
        run.current_node = done_node
        run.phase_start_ms = now_ms
        return old_phase

    def _find_node_by_phase(self, phase_id: PhaseId) -> str | None:
        for name, node in self._graph.nodes.items():
            if node.phase_id == phase_id:
                return name
        return None
