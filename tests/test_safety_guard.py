"""Tests for SafetyGuard: check, clamp, check_hint_freshness."""

import pytest

from halo.contracts.actions import ZERO_ACTION, Action
from halo.contracts.enums import SafetyReflexReason
from halo.contracts.snapshots import TargetInfo
from halo.services.control_service.config import ControlServiceConfig
from halo.services.control_service.safety_guard import SafetyGuard


@pytest.fixture
def cfg() -> ControlServiceConfig:
    return ControlServiceConfig(
        max_linear_delta_m=0.01,
        max_angular_delta_rad=0.02,
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


# --- check ---


def test_check_safe_zero_action(guard: SafetyGuard):
    assert guard.check(ZERO_ACTION) == []


def test_check_safe_sub_limit(guard: SafetyGuard):
    a = Action(0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.5)
    assert guard.check(a) == []


def test_check_over_limit_linear_dx(guard: SafetyGuard):
    a = Action(dx=0.02, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, gripper_cmd=0.0)
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


def test_check_over_limit_linear_dy(guard: SafetyGuard):
    a = Action(dx=0.0, dy=0.02, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, gripper_cmd=0.0)
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


def test_check_over_limit_linear_dz(guard: SafetyGuard):
    a = Action(dx=0.0, dy=0.0, dz=0.02, droll=0.0, dpitch=0.0, dyaw=0.0, gripper_cmd=0.0)
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


def test_check_over_limit_angular_droll(guard: SafetyGuard):
    a = Action(dx=0.0, dy=0.0, dz=0.0, droll=0.05, dpitch=0.0, dyaw=0.0, gripper_cmd=0.0)
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


def test_check_over_limit_angular_dpitch(guard: SafetyGuard):
    a = Action(dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.05, dyaw=0.0, gripper_cmd=0.0)
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


def test_check_over_limit_angular_dyaw(guard: SafetyGuard):
    a = Action(dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.05, gripper_cmd=0.0)
    assert SafetyReflexReason.JOINT_LIMIT in guard.check(a)


def test_check_no_duplicate_violations(guard: SafetyGuard):
    """Both linear and angular exceed limits → only one JOINT_LIMIT entry."""
    a = Action(dx=0.05, dy=0.0, dz=0.0, droll=0.1, dpitch=0.0, dyaw=0.0, gripper_cmd=0.0)
    violations = guard.check(a)
    assert violations.count(SafetyReflexReason.JOINT_LIMIT) == 1


# --- clamp ---


def test_clamp_leaves_sub_limit_untouched(guard: SafetyGuard):
    a = Action(0.005, -0.005, 0.003, 0.01, -0.01, 0.01, 0.5)
    assert guard.clamp(a) == a


def test_clamp_linear_to_limit(guard: SafetyGuard, cfg: ControlServiceConfig):
    a = Action(dx=0.1, dy=-0.1, dz=0.1, droll=0.0, dpitch=0.0, dyaw=0.0, gripper_cmd=0.5)
    clamped = guard.clamp(a)
    assert clamped.dx == pytest.approx(cfg.max_linear_delta_m)
    assert clamped.dy == pytest.approx(-cfg.max_linear_delta_m)
    assert clamped.dz == pytest.approx(cfg.max_linear_delta_m)


def test_clamp_angular_to_limit(guard: SafetyGuard, cfg: ControlServiceConfig):
    a = Action(0.0, 0.0, 0.0, 0.5, -0.5, 0.5, 0.0)
    clamped = guard.clamp(a)
    assert clamped.droll == pytest.approx(cfg.max_angular_delta_rad)
    assert clamped.dpitch == pytest.approx(-cfg.max_angular_delta_rad)
    assert clamped.dyaw == pytest.approx(cfg.max_angular_delta_rad)


def test_clamp_gripper_below_zero(guard: SafetyGuard):
    a = Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5)
    assert guard.clamp(a).gripper_cmd == pytest.approx(0.0)


def test_clamp_gripper_above_one(guard: SafetyGuard):
    a = Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5)
    assert guard.clamp(a).gripper_cmd == pytest.approx(1.0)


# --- check_hint_freshness ---


def test_freshness_true_when_target_none(guard: SafetyGuard, cfg: ControlServiceConfig):
    assert guard.check_hint_freshness(None, cfg) is True


def test_freshness_true_when_hint_valid_and_age_ok(guard: SafetyGuard, cfg: ControlServiceConfig):
    t = _target(hint_valid=True, obs_age_ms=100)
    assert guard.check_hint_freshness(t, cfg) is True


def test_freshness_false_when_hint_invalid(guard: SafetyGuard, cfg: ControlServiceConfig):
    t = _target(hint_valid=False, obs_age_ms=50)
    assert guard.check_hint_freshness(t, cfg) is False


def test_freshness_false_when_obs_age_at_threshold(guard: SafetyGuard, cfg: ControlServiceConfig):
    # obs_age_ms == max_obs_age_ms → not strictly less than → False
    t = _target(hint_valid=True, obs_age_ms=200)
    assert guard.check_hint_freshness(t, cfg) is False


def test_freshness_false_when_obs_age_exceeds_threshold(guard: SafetyGuard, cfg: ControlServiceConfig):
    t = _target(hint_valid=True, obs_age_ms=300)
    assert guard.check_hint_freshness(t, cfg) is False
