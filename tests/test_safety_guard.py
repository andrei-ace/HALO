"""Tests for SafetyGuard: joint-limit check, velocity check, clamp, check_hint_freshness."""

import pytest

from halo.contracts.actions import ZERO_JOINT_ACTION, JointPositionAction
from halo.contracts.enums import SafetyReflexReason
from halo.contracts.snapshots import TargetInfo
from halo.services.control_service.config import ControlServiceConfig
from halo.services.control_service.safety_guard import SafetyGuard


@pytest.fixture
def cfg() -> ControlServiceConfig:
    return ControlServiceConfig(
        max_obs_age_ms=200,
    )


@pytest.fixture
def guard(cfg: ControlServiceConfig) -> SafetyGuard:
    return SafetyGuard(cfg)


def _target(hint_valid: bool = True, obs_age_ms: int = 100) -> TargetInfo:
    return TargetInfo(
        handle="obj-1",
        hint_valid=hint_valid,
        confidence=0.9,
        obs_age_ms=obs_age_ms,
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, 0.0),
        distance_m=0.5,
    )


# --- check: absolute joint limits ---


def test_check_safe_zero_action(guard: SafetyGuard):
    assert guard.check(ZERO_JOINT_ACTION) == []


def test_check_safe_within_limits(guard: SafetyGuard):
    a = JointPositionAction(values=(0.5, -0.5, 0.3, 0.1, 0.2, 1.0))
    assert guard.check(a) == []


def test_check_out_of_range_shoulder_pan(guard: SafetyGuard):
    # shoulder_pan limit is ±1.92; 3.0 is out of range
    a = JointPositionAction(values=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


def test_check_out_of_range_shoulder_lift(guard: SafetyGuard):
    # shoulder_lift limit is ±1.75; -2.0 is out of range
    a = JointPositionAction(values=(0.0, -2.0, 0.0, 0.0, 0.0, 0.0))
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


def test_check_out_of_range_gripper(guard: SafetyGuard):
    # gripper upper limit is 1.75; 2.0 is out of range
    a = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 2.0))
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


def test_check_negative_out_of_range(guard: SafetyGuard):
    # shoulder_pan lower limit is -1.92; -2.5 is out of range
    a = JointPositionAction(values=(-2.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


# --- check: velocity violations are NOT faulted (handled by clamp) ---


def test_check_large_velocity_jump_not_faulted(guard: SafetyGuard):
    """Large delta within absolute limits is NOT faulted — clamp handles it."""
    action = JointPositionAction(values=(1.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert guard.check(action) == []


# --- clamp: absolute limits ---


def test_clamp_within_range_unchanged(guard: SafetyGuard):
    a = JointPositionAction(values=(0.5, -0.5, 0.3, 0.1, 0.2, 1.0))
    clamped = guard.clamp(a)
    assert clamped.values == a.values


def test_clamp_out_of_range(guard: SafetyGuard, cfg: ControlServiceConfig):
    a = JointPositionAction(values=(3.0, -3.0, 0.0, 0.0, 0.0, 2.0))
    clamped = guard.clamp(a)
    assert clamped.values[0] == cfg.joint_limits_upper[0]  # 1.92
    assert clamped.values[1] == cfg.joint_limits_lower[1]  # -1.75
    assert clamped.values[5] == cfg.joint_limits_upper[5]  # 1.75 (gripper)


def test_clamp_all_joints_clamped(guard: SafetyGuard, cfg: ControlServiceConfig):
    """All joints beyond limits get clamped to their respective bounds."""
    a = JointPositionAction(values=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0))
    clamped = guard.clamp(a)
    for i in range(6):
        assert clamped.values[i] == pytest.approx(cfg.joint_limits_upper[i])


# --- clamp: per-tick velocity limit ---


def test_clamp_velocity_limits_large_jump(guard: SafetyGuard):
    """Large jump is clamped to max_joint_delta_rad from prev."""
    prev = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    action = JointPositionAction(values=(1.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    clamped = guard.clamp(action, prev)
    assert clamped.values[0] == pytest.approx(0.5)  # max_joint_delta_rad


def test_clamp_velocity_limits_negative_jump(guard: SafetyGuard):
    """Large negative jump is clamped to -max_joint_delta_rad from prev."""
    prev = JointPositionAction(values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    action = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    clamped = guard.clamp(action, prev)
    assert clamped.values[0] == pytest.approx(0.5)  # 1.0 - 0.5


def test_clamp_velocity_no_effect_without_prev(guard: SafetyGuard):
    """Without prev, only absolute clamp applies."""
    action = JointPositionAction(values=(1.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    clamped = guard.clamp(action)
    assert clamped.values[0] == pytest.approx(1.5)


def test_clamp_gripper_uses_max_gripper_delta():
    """Gripper velocity clamp uses max_gripper_delta, not max_joint_delta_rad."""
    cfg = ControlServiceConfig(max_joint_delta_rad=0.1, max_gripper_delta=1.0)
    guard = SafetyGuard(cfg)
    prev = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    # Gripper jumps 0.9 — within max_gripper_delta, should not be clamped
    action = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.9))
    clamped = guard.clamp(action, prev)
    assert clamped.values[5] == pytest.approx(0.9)


def test_clamp_gripper_clamped_to_max_gripper_delta():
    """Gripper beyond max_gripper_delta is clamped to it."""
    cfg = ControlServiceConfig(max_joint_delta_rad=0.1, max_gripper_delta=0.5)
    guard = SafetyGuard(cfg)
    prev = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    action = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 1.5))
    clamped = guard.clamp(action, prev)
    assert clamped.values[5] == pytest.approx(0.5)


# --- clamp: re-clamp after velocity step (out-of-range prev) ---


def test_clamp_reclamps_after_velocity_step_from_out_of_range_prev():
    """When prev is out-of-range, velocity step can produce out-of-range output; re-clamp fixes it."""
    cfg = ControlServiceConfig(max_joint_delta_rad=0.5)
    guard = SafetyGuard(cfg)
    # prev is beyond upper shoulder_pan limit (1.92)
    prev = JointPositionAction(values=(2.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    # Target is in-range but velocity step from 2.5 toward 1.0 → 2.5 - 0.5 = 2.0, still > 1.92
    action = JointPositionAction(values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    clamped = guard.clamp(action, prev)
    # Re-clamp ensures output is within absolute limits
    assert clamped.values[0] == pytest.approx(1.92)


def test_clamp_reclamps_negative_out_of_range_prev():
    """Re-clamp works for negative out-of-range prev too."""
    cfg = ControlServiceConfig(max_joint_delta_rad=0.5)
    guard = SafetyGuard(cfg)
    prev = JointPositionAction(values=(-2.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    action = JointPositionAction(values=(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    clamped = guard.clamp(action, prev)
    # -2.5 + 0.5 = -2.0, still < -1.92 → re-clamp to -1.92
    assert clamped.values[0] == pytest.approx(-1.92)


# --- check_hint_freshness ---


def test_freshness_true_when_target_none(guard: SafetyGuard):
    assert guard.check_hint_freshness(None) is True


def test_freshness_true_when_hint_valid_and_age_ok(guard: SafetyGuard):
    t = _target(hint_valid=True, obs_age_ms=100)
    assert guard.check_hint_freshness(t) is True


def test_freshness_false_when_hint_invalid(guard: SafetyGuard):
    t = _target(hint_valid=False, obs_age_ms=50)
    assert guard.check_hint_freshness(t) is False


def test_freshness_false_when_obs_age_at_threshold(guard: SafetyGuard):
    # obs_age_ms == max_obs_age_ms → not strictly less than → False
    t = _target(hint_valid=True, obs_age_ms=200)
    assert guard.check_hint_freshness(t) is False


def test_freshness_false_when_obs_age_exceeds_threshold(guard: SafetyGuard):
    t = _target(hint_valid=True, obs_age_ms=300)
    assert guard.check_hint_freshness(t) is False
