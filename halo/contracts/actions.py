from __future__ import annotations

from dataclasses import dataclass

from halo.contracts.enums import PhaseId


@dataclass(frozen=True)
class Action:
    """Per-timestep EE-frame delta command."""

    dx: float
    dy: float
    dz: float
    droll: float
    dpitch: float
    dyaw: float
    gripper_cmd: float  # 0.0=open  1.0=close


ZERO_ACTION = Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # safe hold


@dataclass(frozen=True)
class ActionChunk:
    chunk_id: str
    arm_id: str
    phase_id: PhaseId
    actions: tuple[Action, ...]  # ordered, oldest first
    ts_ms: int  # generation timestamp


@dataclass(frozen=True)
class JointPositionAction:
    """Joint-position targets (radians). Last element is gripper."""

    values: tuple[float, ...]  # (N_DOF,) e.g. (6,) for SO-101


ZERO_JOINT_ACTION = JointPositionAction(values=(0.0,) * 6)


@dataclass(frozen=True)
class JointPositionChunk:
    chunk_id: str
    arm_id: str
    phase_id: PhaseId
    actions: tuple[JointPositionAction, ...]
    ts_ms: int
