"""Tests for JointSafetyGuard: joint-limit checks + clamping."""

from halo.contracts.actions import JointPositionAction
from halo.contracts.enums import SafetyReflexReason
from halo.services.control_service.config import JointControlConfig
from halo.services.control_service.joint_safety_guard import JointSafetyGuard


def _cfg(**kwargs) -> JointControlConfig:
    return JointControlConfig(**kwargs)


def _guard(cfg: JointControlConfig | None = None) -> JointSafetyGuard:
    return JointSafetyGuard(cfg or _cfg())


# --- check() ---


def test_in_range_passes():
    guard = _guard()
    action = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.5))
    assert guard.check(action) == []


def test_out_of_range_flags_joint_limit():
    guard = _guard()
    # shoulder_pan limit is ±1.92; 3.0 is out of range
    action = JointPositionAction(values=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    violations = guard.check(action)
    assert violations == [SafetyReflexReason.JOINT_LIMIT]


def test_gripper_out_of_range_flags_joint_limit():
    guard = _guard()
    # gripper upper limit is 1.75; 2.0 is out of range
    action = JointPositionAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 2.0))
    violations = guard.check(action)
    assert violations == [SafetyReflexReason.JOINT_LIMIT]


def test_negative_out_of_range():
    guard = _guard()
    # shoulder_pan lower limit is -1.92; -2.5 is out of range
    action = JointPositionAction(values=(-2.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    violations = guard.check(action)
    assert violations == [SafetyReflexReason.JOINT_LIMIT]


# --- clamp() ---


def test_clamp_within_range_unchanged():
    guard = _guard()
    action = JointPositionAction(values=(0.5, -0.5, 0.3, 0.1, 0.2, 1.0))
    clamped = guard.clamp(action)
    assert clamped.values == action.values


def test_clamp_out_of_range():
    guard = _guard()
    action = JointPositionAction(values=(3.0, -3.0, 0.0, 0.0, 0.0, 2.0))
    clamped = guard.clamp(action)
    cfg = _cfg()
    assert clamped.values[0] == cfg.joint_limits_upper[0]  # 1.92
    assert clamped.values[1] == cfg.joint_limits_lower[1]  # -1.75
    assert clamped.values[5] == cfg.joint_limits_upper[5]  # 1.75 (gripper)
