from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from halo.contracts.enums import (
    PerceptionFailureCode,
    SkillFailureCode,
    TrackingStatus,
)
from halo.contracts.snapshots import ActInfo, PerceptionInfo, TargetInfo
from halo.services.skill_runner_service.config import SkillRunnerConfig


@dataclass
class StateContext:
    now_ms: int
    elapsed_ms: int
    target: TargetInfo | None
    perception: PerceptionInfo
    act: ActInfo
    config: SkillRunnerConfig
    state_bag: dict
    target_handle: str
    successors: tuple[str, ...]
    held_object_handle: str | None = None


@dataclass(frozen=True)
class HandlerResult:
    transition_to: str | None = None
    fail_code: SkillFailureCode | None = None
    succeed: bool = False
    trigger: str = ""

    @staticmethod
    def stay() -> HandlerResult:
        return HandlerResult()

    @staticmethod
    def go(node: str, trigger: str = "") -> HandlerResult:
        return HandlerResult(transition_to=node, trigger=trigger)

    @staticmethod
    def fail(code: SkillFailureCode, trigger: str = "") -> HandlerResult:
        return HandlerResult(fail_code=code, trigger=trigger)

    @staticmethod
    def done(trigger: str = "success") -> HandlerResult:
        return HandlerResult(succeed=True, trigger=trigger)


class StateHandler(Protocol):
    def evaluate(self, ctx: StateContext) -> HandlerResult: ...


class GlobalGuard(Protocol):
    def check(self, ctx: StateContext) -> HandlerResult | None: ...


# --- Built-in handlers ---


class PassThroughHandler:
    """Immediately transitions to the first successor."""

    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.successors:
            return HandlerResult.go(ctx.successors[0], trigger="pass_through")
        return HandlerResult.stay()


class TimerHandler:
    """Transitions to first successor after a duration stored in config."""

    def __init__(self, duration_ms_key: str, target_node: str | None = None) -> None:
        self._duration_ms_key = duration_ms_key
        self._target_node = target_node

    def evaluate(self, ctx: StateContext) -> HandlerResult:
        duration_ms = getattr(ctx.config, self._duration_ms_key)
        if ctx.elapsed_ms >= duration_ms:
            target = self._target_node or (ctx.successors[0] if ctx.successors else None)
            if target:
                return HandlerResult.go(target, trigger="timer_elapsed")
        return HandlerResult.stay()


# --- Global guards ---


class ReacquireFailedGuard:
    def check(self, ctx: StateContext) -> HandlerResult | None:
        if ctx.perception.failure_code == PerceptionFailureCode.REACQUIRE_FAILED:
            return HandlerResult.fail(SkillFailureCode.PERCEPTION_LOST, trigger="reacquire_failed")
        return None


class PlaceHeldObjectGuard:
    """Fail PLACE immediately if no object is held."""

    def check(self, ctx: StateContext) -> HandlerResult | None:
        if not ctx.held_object_handle:
            return HandlerResult.fail(SkillFailureCode.PLACE_MISS, trigger="no_held_object")
        return None


# --- PICK handlers ---


def _check_target_loss(ctx: StateContext) -> HandlerResult | None:
    """Shared target-loss logic for MOVE_PREGRASP, VISUAL_ALIGN, EXECUTE_APPROACH."""
    target_ok = ctx.target is not None and ctx.target.hint_valid
    if not target_ok:
        no_target_start = ctx.state_bag.get("no_target_start_ms")
        if no_target_start is None:
            ctx.state_bag["no_target_start_ms"] = ctx.now_ms
            return HandlerResult.stay()
        if ctx.now_ms - no_target_start >= ctx.config.no_target_tolerance_ms:
            return HandlerResult.go("RECOVER_RETRY_APPROACH", trigger="target_lost")
        return HandlerResult.stay()
    ctx.state_bag["no_target_start_ms"] = None
    return None


class SelectGraspHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.select_grasp_timeout_ms:
            return HandlerResult.fail(SkillFailureCode.PERCEPTION_LOST, trigger="timeout")
        if ctx.perception.tracking_status != TrackingStatus.TRACKING:
            return HandlerResult.stay()
        if ctx.target is None or ctx.target.handle != ctx.target_handle:
            return HandlerResult.stay()
        return HandlerResult.go("PLAN_APPROACH", trigger="tracking_ok")


class PlanApproachHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.plan_approach_timeout_ms:
            return HandlerResult.fail(SkillFailureCode.NO_PROGRESS, trigger="timeout")
        return HandlerResult.go("MOVE_PREGRASP", trigger="pass_through")


class MovePregraspHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.move_pregrasp_timeout_ms:
            return HandlerResult.fail(SkillFailureCode.NO_PROGRESS, trigger="timeout")
        loss = _check_target_loss(ctx)
        if loss is not None:
            return loss
        if ctx.target is not None and ctx.target.distance_m < ctx.config.approach_align_threshold_m:
            return HandlerResult.go("VISUAL_ALIGN", trigger="close_enough")
        return HandlerResult.stay()


class VisualAlignHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.visual_align_timeout_ms:
            return HandlerResult.fail(SkillFailureCode.NO_PROGRESS, trigger="timeout")
        loss = _check_target_loss(ctx)
        if loss is not None:
            return loss
        if ctx.target is not None and ctx.target.distance_m < ctx.config.execute_approach_threshold_m:
            return HandlerResult.go("EXECUTE_APPROACH", trigger="close_enough")
        return HandlerResult.stay()


class ExecuteApproachHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.execute_approach_timeout_ms:
            ctx.state_bag["grasp_qualify_start_ms"] = None
            return HandlerResult.fail(SkillFailureCode.NO_GRASP, trigger="timeout")
        loss = _check_target_loss(ctx)
        if loss is not None:
            return loss
        if ctx.target is not None and ctx.target.distance_m < ctx.config.grasp_distance_threshold_m:
            qualify_start = ctx.state_bag.get("grasp_qualify_start_ms")
            if qualify_start is None:
                ctx.state_bag["grasp_qualify_start_ms"] = ctx.now_ms
            if (
                ctx.state_bag.get("grasp_qualify_start_ms") is not None
                and ctx.now_ms - ctx.state_bag["grasp_qualify_start_ms"] >= ctx.config.grasp_persistence_ms
            ):
                ctx.state_bag["grasp_qualify_start_ms"] = None
                return HandlerResult.go("CLOSE_GRIPPER", trigger="grasp_qualified")
        else:
            ctx.state_bag["grasp_qualify_start_ms"] = None
        return HandlerResult.stay()


class CloseGripperHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.close_gripper_duration_ms:
            return HandlerResult.go("LIFT", trigger="timer_elapsed")
        return HandlerResult.stay()


class LiftHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.lift_duration_ms:
            if ctx.config.skip_verify_grasp:
                return HandlerResult.done(trigger="skip_verify")
            return HandlerResult.go("VERIFY_GRASP", trigger="lift_complete")
        return HandlerResult.stay()


class VerifyGraspHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.verify_duration_ms:
            return HandlerResult.done(trigger="success")
        return HandlerResult.stay()


class RecoverRetryApproachHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.recover_wait_ms:
            ctx.state_bag["reacquire_count"] = ctx.state_bag.get("reacquire_count", 0) + 1
            if ctx.state_bag["reacquire_count"] > ctx.config.max_reacquire_attempts:
                return HandlerResult.fail(SkillFailureCode.TIMEOUT, trigger="max_retries")
            ctx.state_bag["no_target_start_ms"] = None
            return HandlerResult.go("MOVE_PREGRASP", trigger="retry")
        return HandlerResult.stay()


# --- TRACK handlers ---


class AcquiringHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        timeout_ms = ctx.config.acquiring_timeout_ms * max(1, ctx.config.acquiring_retry_budget)
        if ctx.elapsed_ms >= timeout_ms:
            if (
                ctx.perception.tracking_status == TrackingStatus.TRACKING
                and ctx.target is not None
                and ctx.target.handle != ctx.target_handle
            ):
                return HandlerResult.fail(SkillFailureCode.TARGET_MISMATCH, trigger="timeout")
            return HandlerResult.fail(SkillFailureCode.PERCEPTION_LOST, trigger="timeout")
        if ctx.perception.tracking_status != TrackingStatus.TRACKING:
            return HandlerResult.stay()
        if ctx.target is None or ctx.target.handle != ctx.target_handle:
            return HandlerResult.stay()
        return HandlerResult.done(trigger="tracking_ok")


# --- PLACE handlers ---


class SelectPlaceHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.select_place_timeout_ms:
            if ctx.perception.tracking_status == TrackingStatus.IDLE:
                return HandlerResult.fail(
                    SkillFailureCode.PERCEPTION_LOST,
                    trigger="place_target_not_tracked",
                )
            if (
                ctx.perception.tracking_status == TrackingStatus.TRACKING
                and ctx.target is not None
                and ctx.target.handle != ctx.target_handle
            ):
                return HandlerResult.fail(
                    SkillFailureCode.PERCEPTION_LOST,
                    trigger="tracking_wrong_target",
                )
            return HandlerResult.fail(SkillFailureCode.PERCEPTION_LOST, trigger="timeout")
        if ctx.perception.tracking_status != TrackingStatus.TRACKING:
            return HandlerResult.stay()
        if ctx.target is None or ctx.target.handle != ctx.target_handle:
            return HandlerResult.stay()
        return HandlerResult.go("TRANSIT_PREPLACE", trigger="tracking_ok")


class TransitPreplaceHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.transit_preplace_timeout_ms:
            return HandlerResult.fail(SkillFailureCode.NO_PROGRESS, trigger="timeout")
        loss = _check_target_loss(ctx)
        if loss is not None:
            return loss
        if ctx.target is not None and ctx.target.distance_m < ctx.config.place_align_threshold_m:
            return HandlerResult.go("DESCEND_PLACE", trigger="close_enough")
        return HandlerResult.stay()


class DescendPlaceHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.descend_place_timeout_ms:
            return HandlerResult.fail(SkillFailureCode.PLACE_MISS, trigger="timeout")
        loss = _check_target_loss(ctx)
        if loss is not None:
            return loss
        if ctx.target is not None and ctx.target.distance_m < ctx.config.place_distance_threshold_m:
            return HandlerResult.go("OPEN", trigger="close_enough")
        return HandlerResult.stay()


class OpenHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.open_gripper_duration_ms:
            return HandlerResult.go("RETREAT", trigger="timer_elapsed")
        return HandlerResult.stay()


class RetreatHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.retreat_duration_ms:
            return HandlerResult.go("RETURNING", trigger="trajectory_complete")
        return HandlerResult.stay()


class ReturningHandler:
    """Waits for sim to complete the return trajectory, then transitions to DONE."""

    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.returning_timeout_ms:
            return HandlerResult.done(trigger="trajectory_complete")
        return HandlerResult.stay()


class PlaceRecoverRetryHandler:
    def evaluate(self, ctx: StateContext) -> HandlerResult:
        if ctx.elapsed_ms >= ctx.config.recover_wait_ms:
            ctx.state_bag["reacquire_count"] = ctx.state_bag.get("reacquire_count", 0) + 1
            if ctx.state_bag["reacquire_count"] > ctx.config.max_reacquire_attempts:
                return HandlerResult.fail(SkillFailureCode.TIMEOUT, trigger="max_retries")
            ctx.state_bag["no_target_start_ms"] = None
            return HandlerResult.go("TRANSIT_PREPLACE", trigger="retry")
        return HandlerResult.stay()


# --- Handler factories ---


def build_pick_handlers() -> dict[str, StateHandler]:
    return {
        "SELECT_GRASP": SelectGraspHandler(),
        "PLAN_APPROACH": PlanApproachHandler(),
        "MOVE_PREGRASP": MovePregraspHandler(),
        "VISUAL_ALIGN": VisualAlignHandler(),
        "EXECUTE_APPROACH": ExecuteApproachHandler(),
        "CLOSE_GRIPPER": CloseGripperHandler(),
        "VERIFY_GRASP": VerifyGraspHandler(),
        "LIFT": LiftHandler(),
        "RETURNING": ReturningHandler(),
        "RECOVER_RETRY_APPROACH": RecoverRetryApproachHandler(),
    }


def build_track_handlers() -> dict[str, StateHandler]:
    return {
        "ACQUIRING": AcquiringHandler(),
    }


def build_place_handlers() -> dict[str, StateHandler]:
    return {
        "SELECT_PLACE": SelectPlaceHandler(),
        "TRANSIT_PREPLACE": TransitPreplaceHandler(),
        "DESCEND_PLACE": DescendPlaceHandler(),
        "OPEN": OpenHandler(),
        "RETREAT": RetreatHandler(),
        "RETURNING": ReturningHandler(),
        "RECOVER_RETRY_APPROACH": PlaceRecoverRetryHandler(),
    }


def build_pick_global_guards() -> list[GlobalGuard]:
    return [ReacquireFailedGuard()]


def build_track_global_guards() -> list[GlobalGuard]:
    return []


def build_place_global_guards() -> list[GlobalGuard]:
    return [PlaceHeldObjectGuard(), ReacquireFailedGuard()]
