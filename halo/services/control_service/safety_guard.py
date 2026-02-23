from __future__ import annotations

from halo.contracts.actions import Action
from halo.contracts.enums import SafetyReflexReason
from halo.contracts.snapshots import TargetInfo
from halo.services.control_service.config import ControlServiceConfig


class SafetyGuard:
    """
    Hard guards and clamps on action deltas only (v0: no absolute workspace check).
    Violations list is empty → action is safe to apply.
    """

    def __init__(self, config: ControlServiceConfig) -> None:
        self._config = config

    def check(self, action: Action) -> list[SafetyReflexReason]:
        """
        Returns [JOINT_LIMIT] if any linear delta > max_linear_delta_m
        or any angular delta > max_angular_delta_rad. Returns [] if safe.
        """
        max_l = self._config.max_linear_delta_m
        max_a = self._config.max_angular_delta_rad

        linear_violation = (
            abs(action.dx) > max_l
            or abs(action.dy) > max_l
            or abs(action.dz) > max_l
        )
        angular_violation = (
            abs(action.droll) > max_a
            or abs(action.dpitch) > max_a
            or abs(action.dyaw) > max_a
        )

        if linear_violation or angular_violation:
            return [SafetyReflexReason.JOINT_LIMIT]
        return []

    def clamp(self, action: Action) -> Action:
        """
        Clamp each delta to configured limits; gripper to [0, 1].
        Always returns a valid action (never raises).
        """
        max_l = self._config.max_linear_delta_m
        max_a = self._config.max_angular_delta_rad

        def cl(v: float, limit: float) -> float:
            return max(-limit, min(limit, v))

        return Action(
            dx=cl(action.dx, max_l),
            dy=cl(action.dy, max_l),
            dz=cl(action.dz, max_l),
            droll=cl(action.droll, max_a),
            dpitch=cl(action.dpitch, max_a),
            dyaw=cl(action.dyaw, max_a),
            gripper_cmd=max(0.0, min(1.0, action.gripper_cmd)),
        )

    def check_hint_freshness(
        self, target: TargetInfo | None, config: ControlServiceConfig
    ) -> bool:
        """
        Returns True (fresh) if target is None OR (hint_valid AND obs_age_ms < max_obs_age_ms).
        Returns False → ControlService should hold position.
        """
        if target is None:
            return True
        return target.hint_valid and target.obs_age_ms < config.max_obs_age_ms
