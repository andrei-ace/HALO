"""ZeroMQ server — runs inside Isaac Sim's Python.

Single-env mode for HALO bridge (not batched — HALO drives one arm at a time).
Batched mode is for teacher demo generation only.

Loop:
  1. recv action msg (ZMQ REP)
  2. env.step(action)
  3. build obs from env state
  4. send obs + step_ack (ZMQ REP)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import zmq

from halo_sim.bridge.protocol import (
    MSG_ACTION,
    MSG_RESET,
    ActionMessage,
    ObservationMessage,
    ResetMessage,
    StepAckMessage,
    decode,
    encode,
    wrist_rgb_to_bytes,
)
from halo_sim.constants import PHASE_IDLE, WRIST_ACTIVE_PHASES

logger = logging.getLogger(__name__)


class SimServer:
    """ZeroMQ REP server bridging HALO to a single Isaac Lab environment."""

    def __init__(
        self,
        env: Any,  # PickEnv instance
        action_port: int = 5555,
        obs_port: int = 5556,
        arm_id: str = "arm0",
    ) -> None:
        self._env = env
        self._arm_id = arm_id
        self._action_port = action_port
        self._obs_port = obs_port
        self._context: zmq.Context | None = None
        self._action_socket: zmq.Socket | None = None
        self._obs_socket: zmq.Socket | None = None
        self._current_phase_id: int = PHASE_IDLE

    def start(self) -> None:
        """Bind ZMQ sockets."""
        self._context = zmq.Context()
        self._action_socket = self._context.socket(zmq.REP)
        self._action_socket.bind(f"tcp://*:{self._action_port}")
        self._obs_socket = self._context.socket(zmq.PUB)
        self._obs_socket.bind(f"tcp://*:{self._obs_port}")
        logger.info("SimServer started on action=%d obs=%d", self._action_port, self._obs_port)

    def stop(self) -> None:
        """Close sockets and context."""
        if self._action_socket is not None:
            self._action_socket.close()
        if self._obs_socket is not None:
            self._obs_socket.close()
        if self._context is not None:
            self._context.term()
        logger.info("SimServer stopped")

    def run(self) -> None:
        """Main loop — blocks until interrupted."""
        self.start()
        try:
            while True:
                self._handle_one_message()
        except KeyboardInterrupt:
            logger.info("SimServer interrupted")
        finally:
            self.stop()

    def _handle_one_message(self) -> None:
        """Receive one message, process it, send reply."""
        assert self._action_socket is not None
        raw = self._action_socket.recv()
        msg = decode(raw)
        msg_type = msg.get("type")

        if msg_type == MSG_ACTION:
            action_msg = ActionMessage.from_dict(msg)
            reply = self._handle_action(action_msg)
        elif msg_type == MSG_RESET:
            reset_msg = ResetMessage.from_dict(msg)
            reply = self._handle_reset(reset_msg)
        else:
            reply = {"type": "error", "message": f"unknown message type: {msg_type}"}

        self._action_socket.send(encode(reply))

    def _handle_action(self, msg: ActionMessage) -> dict:
        """Step the environment with the given action."""
        # Update phase from HALO — drives wrist camera gating on the sim side
        self._current_phase_id = msg.phase_id

        action = np.array(msg.action, dtype=np.float32).reshape(1, -1)

        # Step environment (single env, index 0)
        self._env._pre_physics_step(action)
        # In real Isaac Lab: self._env.sim.step()
        self._env._step_count += 1

        # Update env phase for wrist camera gating, then build observation
        self._env.set_phase_ids(np.array([self._current_phase_id], dtype=np.int32))
        obs = self._env._get_observations()
        terminated, truncated = self._env._get_dones()
        done = bool(terminated[0] or truncated[0])
        success = bool(terminated[0])

        # Build observation message from env state (single env, index 0)
        wrist_enabled = bool(self._current_phase_id in WRIST_ACTIVE_PHASES)
        obs_msg = ObservationMessage(
            arm_id=self._arm_id,
            wrist_rgb=wrist_rgb_to_bytes(obs["wrist_rgb"][0]),
            joint_pos=obs["joint_pos"][0].tolist(),
            gripper_state=float(obs["gripper_state"][0]),
            ee_pos=obs["ee_pos"][0].tolist(),
            ee_quat=obs["ee_quat"][0].tolist(),
            cube_pos=obs["cube_pos"][0].tolist(),
            wrist_enabled=wrist_enabled,
            ts_ms=int(time.time() * 1000),
        )

        # Publish observation on PUB socket
        if self._obs_socket is not None:
            self._obs_socket.send(encode(obs_msg.to_dict()))

        # Reply with step ack
        ack = StepAckMessage(done=done, success=success, phase_id=self._current_phase_id)
        return ack.to_dict()

    def _handle_reset(self, msg: ResetMessage) -> dict:
        """Reset the environment."""
        self._env._reset_idx(np.array([0]))
        self._current_phase_id = PHASE_IDLE
        self._env.set_phase_ids(np.array([PHASE_IDLE], dtype=np.int32))

        obs = self._env._get_observations()
        obs_msg = ObservationMessage(
            arm_id=self._arm_id,
            wrist_rgb=wrist_rgb_to_bytes(obs["wrist_rgb"][0]),
            joint_pos=obs["joint_pos"][0].tolist(),
            gripper_state=float(obs["gripper_state"][0]),
            ee_pos=obs["ee_pos"][0].tolist(),
            ee_quat=obs["ee_quat"][0].tolist(),
            cube_pos=obs["cube_pos"][0].tolist(),
            wrist_enabled=False,
            ts_ms=int(time.time() * 1000),
        )

        if self._obs_socket is not None:
            self._obs_socket.send(encode(obs_msg.to_dict()))

        ack = StepAckMessage(done=False, success=False, phase_id=PHASE_IDLE)
        return ack.to_dict()
