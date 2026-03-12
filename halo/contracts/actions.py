from __future__ import annotations

from dataclasses import dataclass

from halo.contracts.enums import PhaseId

SO101_DOF = 6  # 5 arm joints + 1 gripper


@dataclass(frozen=True)
class JointPositionAction:
    """Joint-position targets (radians). Last element is gripper."""

    values: tuple[float, ...]  # (SO101_DOF,)

    def __post_init__(self) -> None:
        if len(self.values) != SO101_DOF:
            raise ValueError(f"JointPositionAction requires exactly {SO101_DOF} values, got {len(self.values)}")


ZERO_JOINT_ACTION = JointPositionAction(values=(0.0,) * SO101_DOF)


@dataclass(frozen=True)
class JointPositionChunk:
    chunk_id: str
    arm_id: str
    phase_id: PhaseId
    actions: tuple[JointPositionAction, ...]
    ts_ms: int
