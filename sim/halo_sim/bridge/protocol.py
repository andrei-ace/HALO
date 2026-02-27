"""ZeroMQ message protocol for HALO <-> Isaac Lab bridge.

All messages are msgpack-encoded dicts.

Action message (HALO -> Sim):
    {"type": "action", "arm_id": str, "action": [7 floats], "phase_id": int, "ts_ms": int}

Observation message (Sim -> HALO):
    {"type": "obs", "arm_id": str, "wrist_rgb": bytes, "joint_pos": [7 floats],
     "gripper_state": float, "ee_pos": [3 floats], "ee_quat": [4 floats],
     "cube_pos": [3 floats], "wrist_enabled": bool, "ts_ms": int}

Reset message (HALO -> Sim):
    {"type": "reset", "arm_id": str, "seed": int}

Step-ack message (Sim -> HALO):
    {"type": "step_ack", "done": bool, "success": bool, "phase_id": int}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np

# Message type constants
MSG_ACTION = "action"
MSG_OBS = "obs"
MSG_RESET = "reset"
MSG_STEP_ACK = "step_ack"


def encode(msg: dict[str, Any]) -> bytes:
    """Encode a message dict to msgpack bytes."""
    return msgpack.packb(msg, use_bin_type=True)


def decode(data: bytes) -> dict[str, Any]:
    """Decode msgpack bytes to a message dict."""
    return msgpack.unpackb(data, raw=False)


@dataclass
class ActionMessage:
    arm_id: str
    action: list[float]  # 7 floats
    phase_id: int  # current HALO PhaseId — drives wrist camera gating
    ts_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": MSG_ACTION,
            "arm_id": self.arm_id,
            "action": self.action,
            "phase_id": self.phase_id,
            "ts_ms": self.ts_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActionMessage:
        return cls(arm_id=d["arm_id"], action=d["action"], phase_id=d.get("phase_id", 0), ts_ms=d["ts_ms"])


@dataclass
class ObservationMessage:
    arm_id: str
    wrist_rgb: bytes  # raw RGB bytes (H*W*3)
    joint_pos: list[float]  # 7 floats
    gripper_state: float
    ee_pos: list[float]  # 3 floats
    ee_quat: list[float]  # 4 floats
    cube_pos: list[float]  # 3 floats
    wrist_enabled: bool
    ts_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": MSG_OBS,
            "arm_id": self.arm_id,
            "wrist_rgb": self.wrist_rgb,
            "joint_pos": self.joint_pos,
            "gripper_state": self.gripper_state,
            "ee_pos": self.ee_pos,
            "ee_quat": self.ee_quat,
            "cube_pos": self.cube_pos,
            "wrist_enabled": self.wrist_enabled,
            "ts_ms": self.ts_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObservationMessage:
        return cls(
            arm_id=d["arm_id"],
            wrist_rgb=d["wrist_rgb"],
            joint_pos=d["joint_pos"],
            gripper_state=d["gripper_state"],
            ee_pos=d["ee_pos"],
            ee_quat=d["ee_quat"],
            cube_pos=d["cube_pos"],
            wrist_enabled=d["wrist_enabled"],
            ts_ms=d["ts_ms"],
        )


@dataclass
class ResetMessage:
    arm_id: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {"type": MSG_RESET, "arm_id": self.arm_id, "seed": self.seed}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResetMessage:
        return cls(arm_id=d["arm_id"], seed=d["seed"])


@dataclass
class StepAckMessage:
    done: bool
    success: bool
    phase_id: int

    def to_dict(self) -> dict[str, Any]:
        return {"type": MSG_STEP_ACK, "done": self.done, "success": self.success, "phase_id": self.phase_id}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepAckMessage:
        return cls(done=d["done"], success=d["success"], phase_id=d["phase_id"])


def wrist_rgb_to_bytes(rgb: np.ndarray) -> bytes:
    """Convert (H, W, 3) uint8 numpy array to raw bytes."""
    return rgb.tobytes()


def bytes_to_wrist_rgb(data: bytes, height: int, width: int) -> np.ndarray:
    """Convert raw bytes back to (H, W, 3) uint8 numpy array."""
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
