from __future__ import annotations

from halo.contracts.actions import JointPositionAction
from halo.contracts.enums import SafetyReflexReason
from halo.services.control_service.config import JointControlConfig


class JointSafetyGuard:
    """Joint-limit safety checks for joint-position control (teacher mode)."""

    def __init__(self, config: JointControlConfig) -> None:
        self._config = config

    def check(self, action: JointPositionAction) -> list[SafetyReflexReason]:
        """Returns [JOINT_LIMIT] if any joint value is out of range."""
        lower = self._config.joint_limits_lower
        upper = self._config.joint_limits_upper
        for i, v in enumerate(action.values):
            if i < len(lower) and (v < lower[i] or v > upper[i]):
                return [SafetyReflexReason.JOINT_LIMIT]
        return []

    def clamp(self, action: JointPositionAction) -> JointPositionAction:
        """Clamp each joint value to configured limits."""
        lower = self._config.joint_limits_lower
        upper = self._config.joint_limits_upper
        clamped = tuple(max(lower[i], min(upper[i], v)) for i, v in enumerate(action.values) if i < len(lower))
        return JointPositionAction(values=clamped)
