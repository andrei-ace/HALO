from __future__ import annotations

from halo.contracts.enums import WRIST_ACTIVE_PHASES, PhaseId, SkillFailureCode, SkillOutcomeState
from halo.contracts.snapshots import ActInfo, PerceptionInfo, TargetInfo
from halo.services.skill_runner_service.config import SkillRunnerConfig


class PickFSM:
    """Pure (no asyncio) FSM for the Pick skill."""

    def __init__(self, config: SkillRunnerConfig) -> None:
        self._config = config
        self._phase: PhaseId = PhaseId.IDLE
        self._outcome: SkillOutcomeState = SkillOutcomeState.IN_PROGRESS
        self._failure_code: SkillFailureCode | None = None
        self._phase_start_ms: int = 0
        self._grasp_qualify_start_ms: int | None = None
        self._no_target_start_ms: int | None = None
        self._reacquire_count: int = 0

    # --- Readable state ---

    @property
    def phase(self) -> PhaseId:
        return self._phase

    @property
    def outcome(self) -> SkillOutcomeState:
        return self._outcome

    @property
    def failure_code(self) -> SkillFailureCode | None:
        return self._failure_code

    @property
    def phase_start_ms(self) -> int:
        return self._phase_start_ms

    @property
    def is_active(self) -> bool:
        return self._phase != PhaseId.IDLE and self._phase != PhaseId.DONE

    @property
    def wrist_camera_active(self) -> bool:
        return self._phase in WRIST_ACTIVE_PHASES

    # --- Commands ---

    def start(self, now_ms: int) -> None:
        if self._phase != PhaseId.IDLE:
            raise RuntimeError(f"start() called in phase {self._phase!r}, expected IDLE")
        self._grasp_qualify_start_ms = None
        self._no_target_start_ms = None
        self._reacquire_count = 0
        self._outcome = SkillOutcomeState.IN_PROGRESS
        self._failure_code = None
        self._transition(now_ms, PhaseId.SELECT_GRASP)

    def abort(self, now_ms: int) -> None:
        if self._phase == PhaseId.DONE:
            return
        self._failure_code = SkillFailureCode.UNSAFE_ABORT
        self._outcome = SkillOutcomeState.FAILURE
        self._phase = PhaseId.DONE
        self._phase_start_ms = now_ms

    # --- Per-tick ---

    def advance(
        self,
        now_ms: int,
        target: TargetInfo | None,
        perception: PerceptionInfo,
        act: ActInfo,
    ) -> PhaseId | None:
        """Returns old phase if a transition occurred; None if stayed."""
        if not self.is_active:
            return None

        phase = self._phase
        elapsed = now_ms - self._phase_start_ms

        if phase == PhaseId.SELECT_GRASP:
            if elapsed >= self._config.select_grasp_timeout_ms:
                return self._fail(now_ms, SkillFailureCode.NO_PROGRESS)
            # v0: immediate pass-through (grasp planning is a future extension)
            return self._transition(now_ms, PhaseId.PLAN_APPROACH)

        elif phase == PhaseId.PLAN_APPROACH:
            if elapsed >= self._config.plan_approach_timeout_ms:
                return self._fail(now_ms, SkillFailureCode.NO_PROGRESS)
            # v0: immediate pass-through (motion planning is a future extension)
            return self._transition(now_ms, PhaseId.MOVE_PREGRASP)

        elif phase == PhaseId.MOVE_PREGRASP:
            if elapsed >= self._config.move_pregrasp_timeout_ms:
                return self._fail(now_ms, SkillFailureCode.NO_PROGRESS)
            target_ok = target is not None and target.hint_valid
            if not target_ok:
                if self._no_target_start_ms is None:
                    self._no_target_start_ms = now_ms
                if now_ms - self._no_target_start_ms >= self._config.no_target_tolerance_ms:
                    return self._transition(now_ms, PhaseId.RECOVER_RETRY_APPROACH)
                return None
            self._no_target_start_ms = None
            if target.distance_m < self._config.approach_align_threshold_m:
                return self._transition(now_ms, PhaseId.VISUAL_ALIGN)
            return None

        elif phase == PhaseId.VISUAL_ALIGN:
            if elapsed >= self._config.visual_align_timeout_ms:
                return self._fail(now_ms, SkillFailureCode.NO_PROGRESS)
            target_ok = target is not None and target.hint_valid
            if not target_ok:
                if self._no_target_start_ms is None:
                    self._no_target_start_ms = now_ms
                if now_ms - self._no_target_start_ms >= self._config.no_target_tolerance_ms:
                    return self._transition(now_ms, PhaseId.RECOVER_RETRY_APPROACH)
                return None
            self._no_target_start_ms = None
            if target.distance_m < self._config.execute_approach_threshold_m:
                return self._transition(now_ms, PhaseId.EXECUTE_APPROACH)
            return None

        elif phase == PhaseId.EXECUTE_APPROACH:
            if elapsed >= self._config.execute_approach_timeout_ms:
                self._grasp_qualify_start_ms = None
                return self._fail(now_ms, SkillFailureCode.NO_GRASP)
            target_ok = target is not None and target.hint_valid
            if not target_ok:
                if self._no_target_start_ms is None:
                    self._no_target_start_ms = now_ms
                if now_ms - self._no_target_start_ms >= self._config.no_target_tolerance_ms:
                    return self._transition(now_ms, PhaseId.RECOVER_RETRY_APPROACH)
                return None
            self._no_target_start_ms = None
            if target.distance_m < self._config.grasp_distance_threshold_m:
                if self._grasp_qualify_start_ms is None:
                    self._grasp_qualify_start_ms = now_ms
                if now_ms - self._grasp_qualify_start_ms >= self._config.grasp_persistence_ms:
                    self._grasp_qualify_start_ms = None
                    return self._transition(now_ms, PhaseId.CLOSE_GRIPPER)
            else:
                self._grasp_qualify_start_ms = None
            return None

        elif phase == PhaseId.CLOSE_GRIPPER:
            if elapsed >= self._config.close_gripper_duration_ms:
                if self._config.skip_verify_grasp:
                    return self._transition(now_ms, PhaseId.LIFT)
                else:
                    return self._transition(now_ms, PhaseId.VERIFY_GRASP)
            return None

        elif phase == PhaseId.VERIFY_GRASP:
            if elapsed >= self._config.verify_duration_ms:
                return self._transition(now_ms, PhaseId.LIFT)
            return None

        elif phase == PhaseId.LIFT:
            if elapsed >= self._config.lift_duration_ms:
                return self._transition_success(now_ms)
            return None

        elif phase == PhaseId.RECOVER_RETRY_APPROACH:
            if elapsed >= self._config.recover_wait_ms:
                self._reacquire_count += 1
                if self._reacquire_count > self._config.max_reacquire_attempts:
                    return self._fail(now_ms, SkillFailureCode.TIMEOUT)
                self._no_target_start_ms = None
                return self._transition(now_ms, PhaseId.MOVE_PREGRASP)
            return None

        return None

    def sync_phase(self, now_ms: int, phase_id: PhaseId) -> PhaseId | None:
        """Accept external phase from teacher. Returns old phase on transition, None if no change."""
        if not self.is_active or phase_id == self._phase:
            return None
        if phase_id == PhaseId.DONE:
            return self._transition_success(now_ms)
        return self._transition(now_ms, phase_id)

    def needs_chunk(self, act: ActInfo) -> bool:
        return act.buffer_fill_ms < self._config.buffer_target_ms

    # --- Private helpers ---

    def _transition(self, now_ms: int, new_phase: PhaseId) -> PhaseId:
        old_phase = self._phase
        self._phase = new_phase
        self._phase_start_ms = now_ms
        return old_phase

    def _fail(self, now_ms: int, code: SkillFailureCode) -> PhaseId:
        self._failure_code = code
        self._outcome = SkillOutcomeState.FAILURE
        return self._transition(now_ms, PhaseId.DONE)

    def _transition_success(self, now_ms: int) -> PhaseId:
        self._outcome = SkillOutcomeState.SUCCESS
        return self._transition(now_ms, PhaseId.DONE)
