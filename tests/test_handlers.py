"""Tests for individual state handlers."""

from halo.contracts.enums import (
    ActStatus,
    PerceptionFailureCode,
    SkillFailureCode,
    TrackingStatus,
)
from halo.contracts.snapshots import ActInfo, PerceptionInfo, TargetInfo
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.handlers import (
    AcquiringHandler,
    CloseGripperHandler,
    DescendPlaceHandler,
    ExecuteApproachHandler,
    LiftHandler,
    MovePregraspHandler,
    OpenHandler,
    PlaceRecoverRetryHandler,
    PlanApproachHandler,
    ReacquireFailedGuard,
    RecoverRetryApproachHandler,
    RetreatHandler,
    SelectGraspHandler,
    SelectPlaceHandler,
    StateContext,
    TransitPreplaceHandler,
    VerifyGraspHandler,
)


def _cfg(**kw) -> SkillRunnerConfig:
    return SkillRunnerConfig(**kw)


def _target(distance_m: float = 0.5, hint_valid: bool = True, handle: str = "obj-1") -> TargetInfo:
    return TargetInfo(
        handle=handle,
        hint_valid=hint_valid,
        confidence=0.9,
        obs_age_ms=10,
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, distance_m),
        distance_m=distance_m,
    )


def _perception(status: TrackingStatus = TrackingStatus.TRACKING) -> PerceptionInfo:
    return PerceptionInfo(
        tracking_status=status,
        failure_code=PerceptionFailureCode.OK,
        reacquire_fail_count=0,
        vlm_job_pending=False,
    )


def _act() -> ActInfo:
    return ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=300, buffer_low=False)


def _ctx(
    elapsed_ms: int = 0,
    target: TargetInfo | None = None,
    perception: PerceptionInfo | None = None,
    config: SkillRunnerConfig | None = None,
    state_bag: dict | None = None,
    successors: tuple[str, ...] = ("NEXT",),
    target_handle: str = "obj-1",
    now_ms: int = 1000,
) -> StateContext:
    return StateContext(
        now_ms=now_ms,
        elapsed_ms=elapsed_ms,
        target=target or _target(),
        perception=perception or _perception(),
        act=_act(),
        config=config or _cfg(),
        state_bag=state_bag if state_bag is not None else {},
        target_handle=target_handle,
        successors=successors,
    )


# --- SelectGraspHandler ---


def test_select_grasp_transitions_on_tracking():
    h = SelectGraspHandler()
    result = h.evaluate(_ctx())
    assert result.transition_to == "PLAN_APPROACH"


def test_select_grasp_stays_when_not_tracking():
    h = SelectGraspHandler()
    result = h.evaluate(_ctx(perception=_perception(TrackingStatus.IDLE)))
    assert result.transition_to is None


def test_select_grasp_stays_when_wrong_handle():
    h = SelectGraspHandler()
    result = h.evaluate(_ctx(target=_target(handle="wrong")))
    assert result.transition_to is None


def test_select_grasp_timeout():
    h = SelectGraspHandler()
    result = h.evaluate(_ctx(elapsed_ms=10000, config=_cfg(select_grasp_timeout_ms=10000)))
    assert result.fail_code == SkillFailureCode.PERCEPTION_LOST


# --- PlanApproachHandler ---


def test_plan_approach_passes_through():
    h = PlanApproachHandler()
    result = h.evaluate(_ctx())
    assert result.transition_to == "MOVE_PREGRASP"


# --- MovePregraspHandler ---


def test_move_pregrasp_transitions_on_close():
    h = MovePregraspHandler()
    result = h.evaluate(_ctx(target=_target(distance_m=0.10), config=_cfg(approach_align_threshold_m=0.15)))
    assert result.transition_to == "VISUAL_ALIGN"


def test_move_pregrasp_stays_when_far():
    h = MovePregraspHandler()
    result = h.evaluate(_ctx(target=_target(distance_m=0.5), config=_cfg(approach_align_threshold_m=0.15)))
    assert result.transition_to is None


def test_move_pregrasp_target_loss_triggers_recovery():
    h = MovePregraspHandler()
    bag: dict = {}
    # First tick: sets no_target_start_ms
    h.evaluate(_ctx(target=_target(hint_valid=False), state_bag=bag, config=_cfg(no_target_tolerance_ms=500)))
    # Second tick: tolerance exceeded
    result = h.evaluate(
        _ctx(
            target=_target(hint_valid=False),
            state_bag=bag,
            config=_cfg(no_target_tolerance_ms=500),
            now_ms=1500,
            elapsed_ms=500,
        )
    )
    assert result.transition_to == "RECOVER_RETRY_APPROACH"


# --- ExecuteApproachHandler ---


def test_execute_approach_grasp_qualified():
    h = ExecuteApproachHandler()
    result = h.evaluate(
        _ctx(
            target=_target(distance_m=0.005),
            config=_cfg(grasp_distance_threshold_m=0.01, grasp_persistence_ms=0),
        )
    )
    assert result.transition_to == "CLOSE_GRIPPER"


def test_execute_approach_persistence_reset():
    h = ExecuteApproachHandler()
    bag: dict = {}
    # Below threshold — starts qualifying
    h.evaluate(
        _ctx(
            target=_target(distance_m=0.005),
            state_bag=bag,
            config=_cfg(grasp_distance_threshold_m=0.01, grasp_persistence_ms=500),
        )
    )
    assert bag.get("grasp_qualify_start_ms") is not None
    # Above threshold — resets
    h.evaluate(
        _ctx(
            target=_target(distance_m=0.05),
            state_bag=bag,
            config=_cfg(grasp_distance_threshold_m=0.01, grasp_persistence_ms=500),
        )
    )
    assert bag.get("grasp_qualify_start_ms") is None


# --- CloseGripperHandler ---


def test_close_gripper_to_lift():
    h = CloseGripperHandler()
    result = h.evaluate(_ctx(elapsed_ms=1000, config=_cfg(close_gripper_duration_ms=1000)))
    assert result.transition_to == "LIFT"


# --- LiftHandler ---


def test_lift_to_verify_grasp():
    h = LiftHandler()
    result = h.evaluate(_ctx(elapsed_ms=2000, config=_cfg(lift_duration_ms=2000, skip_verify_grasp=False)))
    assert result.transition_to == "VERIFY_GRASP"


def test_lift_skip_verify():
    h = LiftHandler()
    result = h.evaluate(_ctx(elapsed_ms=2000, config=_cfg(lift_duration_ms=2000, skip_verify_grasp=True)))
    assert result.succeed is True


# --- VerifyGraspHandler ---


def test_verify_grasp_succeeds():
    h = VerifyGraspHandler()
    result = h.evaluate(_ctx(elapsed_ms=500, config=_cfg(verify_duration_ms=500)))
    assert result.succeed is True


# --- RecoverRetryApproachHandler ---


def test_recover_retry():
    h = RecoverRetryApproachHandler()
    bag: dict = {"reacquire_count": 0}
    result = h.evaluate(_ctx(elapsed_ms=500, config=_cfg(recover_wait_ms=500, max_reacquire_attempts=3), state_bag=bag))
    assert result.transition_to == "MOVE_PREGRASP"
    assert bag["reacquire_count"] == 1


def test_recover_max_retries():
    h = RecoverRetryApproachHandler()
    bag: dict = {"reacquire_count": 3}
    result = h.evaluate(_ctx(elapsed_ms=500, config=_cfg(recover_wait_ms=500, max_reacquire_attempts=3), state_bag=bag))
    assert result.fail_code == SkillFailureCode.TIMEOUT


# --- AcquiringHandler ---


def test_acquiring_succeeds():
    h = AcquiringHandler()
    result = h.evaluate(_ctx(target=_target(handle="obj-1"), perception=_perception(TrackingStatus.TRACKING)))
    assert result.succeed is True


def test_acquiring_timeout_perception_lost():
    h = AcquiringHandler()
    result = h.evaluate(
        _ctx(
            elapsed_ms=30000,
            target=None,
            perception=_perception(TrackingStatus.IDLE),
            config=_cfg(acquiring_timeout_ms=10000, acquiring_retry_budget=3),
        )
    )
    assert result.fail_code == SkillFailureCode.PERCEPTION_LOST


def test_acquiring_timeout_target_mismatch():
    h = AcquiringHandler()
    result = h.evaluate(
        _ctx(
            elapsed_ms=30000,
            target=_target(handle="wrong"),
            perception=_perception(TrackingStatus.TRACKING),
            config=_cfg(acquiring_timeout_ms=10000, acquiring_retry_budget=3),
            target_handle="obj-1",
        )
    )
    assert result.fail_code == SkillFailureCode.TARGET_MISMATCH


def test_acquiring_stays_before_retry_budget_exhausted():
    h = AcquiringHandler()
    result = h.evaluate(
        _ctx(
            elapsed_ms=29999,
            target=None,
            perception=_perception(TrackingStatus.REACQUIRING),
            config=_cfg(acquiring_timeout_ms=10000, acquiring_retry_budget=3),
        )
    )
    assert result.transition_to is None
    assert result.fail_code is None


# --- ReacquireFailedGuard ---


def test_reacquire_failed_guard_triggers():
    g = ReacquireFailedGuard()
    perc = PerceptionInfo(
        tracking_status=TrackingStatus.REACQUIRING,
        failure_code=PerceptionFailureCode.REACQUIRE_FAILED,
        reacquire_fail_count=3,
        vlm_job_pending=False,
    )
    result = g.check(_ctx(perception=perc))
    assert result is not None
    assert result.fail_code == SkillFailureCode.PERCEPTION_LOST


def test_reacquire_failed_guard_passes():
    g = ReacquireFailedGuard()
    result = g.check(_ctx())
    assert result is None


# --- SelectPlaceHandler ---


def test_select_place_transitions_on_tracking():
    h = SelectPlaceHandler()
    result = h.evaluate(_ctx())
    assert result.transition_to == "TRANSIT_PREPLACE"


def test_select_place_stays_when_not_tracking():
    h = SelectPlaceHandler()
    result = h.evaluate(_ctx(perception=_perception(TrackingStatus.IDLE)))
    assert result.transition_to is None


def test_select_place_stays_when_wrong_handle():
    h = SelectPlaceHandler()
    result = h.evaluate(_ctx(target=_target(handle="wrong")))
    assert result.transition_to is None


def test_select_place_timeout():
    h = SelectPlaceHandler()
    # TRACKING correct handle but timed out → generic "timeout"
    result = h.evaluate(_ctx(elapsed_ms=10000, config=_cfg(select_place_timeout_ms=10000)))
    assert result.fail_code == SkillFailureCode.PERCEPTION_LOST
    assert result.trigger == "timeout"


def test_select_place_timeout_not_tracked():
    h = SelectPlaceHandler()
    result = h.evaluate(
        _ctx(
            elapsed_ms=30000,
            perception=_perception(TrackingStatus.IDLE),
            config=_cfg(select_place_timeout_ms=30000),
        )
    )
    assert result.fail_code == SkillFailureCode.PERCEPTION_LOST
    assert result.trigger == "place_target_not_tracked"


def test_select_place_timeout_wrong_target():
    h = SelectPlaceHandler()
    result = h.evaluate(
        _ctx(
            elapsed_ms=30000,
            target=_target(handle="green_cube_01"),
            perception=_perception(TrackingStatus.TRACKING),
            config=_cfg(select_place_timeout_ms=30000),
            target_handle="beige_tray_01",
        )
    )
    assert result.fail_code == SkillFailureCode.PERCEPTION_LOST
    assert result.trigger == "tracking_wrong_target"


# --- TransitPreplaceHandler ---


def test_transit_preplace_transitions_on_close():
    h = TransitPreplaceHandler()
    result = h.evaluate(_ctx(target=_target(distance_m=0.05), config=_cfg(place_align_threshold_m=0.10)))
    assert result.transition_to == "DESCEND_PLACE"


def test_transit_preplace_stays_when_far():
    h = TransitPreplaceHandler()
    result = h.evaluate(_ctx(target=_target(distance_m=0.5), config=_cfg(place_align_threshold_m=0.10)))
    assert result.transition_to is None


def test_transit_preplace_timeout():
    h = TransitPreplaceHandler()
    result = h.evaluate(_ctx(elapsed_ms=10000, config=_cfg(transit_preplace_timeout_ms=10000)))
    assert result.fail_code == SkillFailureCode.NO_PROGRESS


def test_transit_preplace_target_loss():
    h = TransitPreplaceHandler()
    bag: dict = {}
    h.evaluate(_ctx(target=_target(hint_valid=False), state_bag=bag, config=_cfg(no_target_tolerance_ms=500)))
    result = h.evaluate(
        _ctx(
            target=_target(hint_valid=False),
            state_bag=bag,
            config=_cfg(no_target_tolerance_ms=500),
            now_ms=1500,
            elapsed_ms=500,
        )
    )
    assert result.transition_to == "RECOVER_RETRY_APPROACH"


# --- DescendPlaceHandler ---


def test_descend_place_transitions_on_close():
    h = DescendPlaceHandler()
    result = h.evaluate(_ctx(target=_target(distance_m=0.01), config=_cfg(place_distance_threshold_m=0.02)))
    assert result.transition_to == "OPEN"


def test_descend_place_stays_when_far():
    h = DescendPlaceHandler()
    result = h.evaluate(_ctx(target=_target(distance_m=0.05), config=_cfg(place_distance_threshold_m=0.02)))
    assert result.transition_to is None


def test_descend_place_timeout():
    h = DescendPlaceHandler()
    result = h.evaluate(_ctx(elapsed_ms=5000, config=_cfg(descend_place_timeout_ms=5000)))
    assert result.fail_code == SkillFailureCode.PLACE_MISS


def test_descend_place_target_loss():
    h = DescendPlaceHandler()
    bag: dict = {}
    h.evaluate(_ctx(target=_target(hint_valid=False), state_bag=bag, config=_cfg(no_target_tolerance_ms=500)))
    result = h.evaluate(
        _ctx(
            target=_target(hint_valid=False),
            state_bag=bag,
            config=_cfg(no_target_tolerance_ms=500),
            now_ms=1500,
            elapsed_ms=500,
        )
    )
    assert result.transition_to == "RECOVER_RETRY_APPROACH"


# --- OpenHandler ---


def test_open_transitions_after_timer():
    h = OpenHandler()
    result = h.evaluate(_ctx(elapsed_ms=1000, config=_cfg(open_gripper_duration_ms=1000)))
    assert result.transition_to == "RETREAT"


def test_open_stays_before_timer():
    h = OpenHandler()
    result = h.evaluate(_ctx(elapsed_ms=500, config=_cfg(open_gripper_duration_ms=1000)))
    assert result.transition_to is None


# --- RetreatHandler ---


def test_retreat_transitions_to_returning_after_timer():
    h = RetreatHandler()
    result = h.evaluate(_ctx(elapsed_ms=2000, config=_cfg(retreat_duration_ms=2000)))
    assert result.transition_to == "RETURNING"


def test_retreat_stays_before_timer():
    h = RetreatHandler()
    result = h.evaluate(_ctx(elapsed_ms=1000, config=_cfg(retreat_duration_ms=2000)))
    assert result.succeed is False
    assert result.transition_to is None


# --- PlaceRecoverRetryHandler ---


def test_place_recover_retry():
    h = PlaceRecoverRetryHandler()
    bag: dict = {"reacquire_count": 0}
    result = h.evaluate(_ctx(elapsed_ms=500, config=_cfg(recover_wait_ms=500, max_reacquire_attempts=3), state_bag=bag))
    assert result.transition_to == "TRANSIT_PREPLACE"
    assert bag["reacquire_count"] == 1


def test_place_recover_max_retries():
    h = PlaceRecoverRetryHandler()
    bag: dict = {"reacquire_count": 3}
    result = h.evaluate(_ctx(elapsed_ms=500, config=_cfg(recover_wait_ms=500, max_reacquire_attempts=3), state_bag=bag))
    assert result.fail_code == SkillFailureCode.TIMEOUT
