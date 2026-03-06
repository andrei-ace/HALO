from __future__ import annotations

from halo.contracts.enums import (
    PhaseId,
    SkillFailureCode,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.snapshots import ActInfo, PerceptionInfo, TargetInfo
from halo.services.skill_runner_service.config import SkillRunnerConfig


class TrackFSM:
    """Pure FSM for the TRACK skill. Gates on tracking_status == TRACKING."""

    def __init__(self, config: SkillRunnerConfig) -> None:
        self._config = config
        self._phase: PhaseId = PhaseId.IDLE
        self._outcome: SkillOutcomeState = SkillOutcomeState.IN_PROGRESS
        self._failure_code: SkillFailureCode | None = None
        self._phase_start_ms: int = 0
        self._target_handle: str | None = None

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

    def start(self, now_ms: int, target_handle: str) -> None:
        if self._phase != PhaseId.IDLE:
            raise RuntimeError(f"start() called in phase {self._phase!r}, expected IDLE")
        self._target_handle = target_handle
        self._outcome = SkillOutcomeState.IN_PROGRESS
        self._failure_code = None
        self._transition(now_ms, PhaseId.ACQUIRING)

    def abort(self, now_ms: int) -> None:
        if self._phase == PhaseId.DONE:
            return
        self._failure_code = SkillFailureCode.UNSAFE_ABORT
        self._outcome = SkillOutcomeState.FAILURE
        self._phase = PhaseId.DONE
        self._phase_start_ms = now_ms

    def fail(self, now_ms: int, code: SkillFailureCode) -> None:
        if self._phase == PhaseId.DONE:
            return
        self._failure_code = code
        self._outcome = SkillOutcomeState.FAILURE
        self._phase = PhaseId.DONE
        self._phase_start_ms = now_ms

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

        if phase == PhaseId.ACQUIRING:
            if elapsed >= self._config.acquiring_timeout_ms:
                # Distinguish: tracker running but wrong handle vs. tracker never locked on
                if (
                    perception.tracking_status == TrackingStatus.TRACKING
                    and target is not None
                    and target.handle != self._target_handle
                ):
                    return self._fail(now_ms, SkillFailureCode.TARGET_MISMATCH)
                return self._fail(now_ms, SkillFailureCode.PERCEPTION_LOST)
            if perception.tracking_status != TrackingStatus.TRACKING:
                return None
            if target is None or target.handle != self._target_handle:
                return None
            return self._transition_success(now_ms)

        return None

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
