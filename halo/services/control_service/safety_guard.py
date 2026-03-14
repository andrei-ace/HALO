from __future__ import annotations

from halo.contracts.actions import JointPositionAction
from halo.contracts.enums import SafetyReflexReason
from halo.contracts.snapshots import TargetInfo
from halo.services.control_service.config import ControlServiceConfig


class SafetyGuard:
    """
    Joint-limit and velocity safety checks for joint-position actions.
    Violations list is empty → action is safe to apply.
    """

    def __init__(self, config: ControlServiceConfig) -> None:
        self._config = config

    def check(
        self,
        action: JointPositionAction,
    ) -> list[SafetyReflexReason]:
        """
        Returns [JOINT_LIMIT] if any joint value is outside the configured
        absolute joint_limits_lower / joint_limits_upper bounds.
        Returns [] if safe.

        Per-tick velocity violations are NOT faulted here — they are handled
        by ``clamp()`` which bounds the step to ``max_joint_delta_rad``.
        This avoids false reflexes on startup/recovery when the first target
        in a chunk is far from the current pose.
        """
        lower = self._config.joint_limits_lower
        upper = self._config.joint_limits_upper
        for i, v in enumerate(action.values):
            if v < lower[i] or v > upper[i]:
                return [SafetyReflexReason.JOINT_LIMIT]

        return []

    def clamp(
        self,
        action: JointPositionAction,
        prev: JointPositionAction | None = None,
    ) -> JointPositionAction:
        """
        Clamp each joint value to configured absolute limits, then enforce
        per-tick velocity limit relative to *prev* (if provided).
        Always returns a valid action (never raises).
        """
        lower = self._config.joint_limits_lower
        upper = self._config.joint_limits_upper
        # 1. Absolute joint-limit clamp
        clamped = [max(lower[i], min(upper[i], v)) for i, v in enumerate(action.values)]

        # 2. Per-tick velocity clamp
        if prev is not None:
            max_delta = self._config.max_joint_delta_rad
            gripper_idx = len(lower) - 1
            needs_reclamp = False
            for i in range(len(clamped)):
                limit = self._config.max_gripper_delta if i == gripper_idx else max_delta
                delta = clamped[i] - prev.values[i]
                if abs(delta) > limit:
                    clamped[i] = prev.values[i] + limit * (1.0 if delta > 0 else -1.0)
                    needs_reclamp = True

            # 3. Re-clamp to absolute limits after velocity step — if prev was
            # out-of-range (e.g. from hardware telemetry), the velocity-limited
            # output can still be outside joint bounds.
            if needs_reclamp:
                for i in range(len(clamped)):
                    clamped[i] = max(lower[i], min(upper[i], clamped[i]))

        return JointPositionAction(values=tuple(clamped))

    def check_hint_freshness(self, target: TargetInfo | None) -> bool:
        """
        Returns True (fresh) if target is None OR (hint_valid AND obs_age_ms < max_obs_age_ms).
        Returns False → ControlService should hold position.
        """
        if target is None:
            return True
        return target.hint_valid and target.obs_age_ms < self._config.max_obs_age_ms
