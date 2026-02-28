"""Command dispatch for CommandRPC (REQ/REP) messages.

Handles: step, reset, get_state, set_state, teacher_step, configure, set_hint, shutdown.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from mujoco_sim.server.protocol import (
    CMD_CONFIGURE,
    CMD_GET_STATE,
    CMD_RESET,
    CMD_SET_HINT,
    CMD_SET_STATE,
    CMD_SHUTDOWN,
    CMD_STEP,
    CMD_TEACHER_STEP,
    RESP_ERROR,
    RESP_OK,
    RESP_RESET_OK,
    RESP_STATE,
    RESP_STEP_OK,
    RESP_TEACHER_STEP_OK,
    ndarray_to_bytes,
)

if TYPE_CHECKING:
    from mujoco_sim.env import SO101Env
    from mujoco_sim.teacher.pick_teacher import PickTeacher

logger = logging.getLogger(__name__)


def dispatch_command(
    msg: dict,
    env: SO101Env,
    teacher: PickTeacher,
    *,
    teacher_mode: bool = False,
) -> tuple[dict, bool]:
    """Dispatch a CommandRPC command and return (response_dict, should_shutdown).

    Args:
        msg: Decoded msgpack command message.
        env: SO101Env instance.
        teacher: PickTeacher instance.
        teacher_mode: If True, teacher_step uses teacher policy.

    Returns:
        (response_dict, should_shutdown) tuple.
    """
    cmd = msg.get("type")

    if cmd == CMD_STEP:
        return _handle_step(msg, env), False

    if cmd == CMD_RESET:
        return _handle_reset(msg, env, teacher), False

    if cmd == CMD_GET_STATE:
        return _handle_get_state(env), False

    if cmd == CMD_SET_STATE:
        return _handle_set_state(msg, env), False

    if cmd == CMD_TEACHER_STEP:
        return _handle_teacher_step(env, teacher, teacher_mode), False

    if cmd == CMD_CONFIGURE:
        return _handle_configure(msg), False

    if cmd == CMD_SET_HINT:
        return _handle_set_hint(msg), False

    if cmd == CMD_SHUTDOWN:
        logger.info("Shutdown command received")
        return {"type": RESP_OK}, True

    return {"type": RESP_ERROR, "message": f"Unknown command: {cmd}"}, False


def _handle_step(msg: dict, env: SO101Env) -> dict:
    """Step the environment with a 6D joint-position action."""
    action_bytes = msg.get("action")
    if action_bytes is None:
        return {"type": RESP_ERROR, "message": "step requires 'action' field"}
    action = np.frombuffer(action_bytes, dtype=np.float64).copy()
    if action.shape != (6,):
        return {"type": RESP_ERROR, "message": f"Expected action shape (6,), got {action.shape}"}

    _obs, reward, done, _info = env.step(action)
    return {
        "type": RESP_STEP_OK,
        "reward": reward,
        "done": done,
    }


def _handle_reset(msg: dict, env: SO101Env, teacher: PickTeacher) -> dict:
    """Reset environment and teacher."""
    seed = msg.get("seed")
    env.reset(seed=seed)
    teacher.reset()
    return {
        "type": RESP_RESET_OK,
        "seed": seed,
    }


def _handle_get_state(env: SO101Env) -> dict:
    """Return full MuJoCo state (qpos, qvel)."""
    state = env.get_state()
    return {
        "type": RESP_STATE,
        "qpos": ndarray_to_bytes(state["qpos"]),
        "qvel": ndarray_to_bytes(state["qvel"]),
    }


def _handle_set_state(msg: dict, env: SO101Env) -> dict:
    """Inject MuJoCo state."""
    qpos = np.frombuffer(msg["qpos"], dtype=np.float64).copy()
    qvel = np.frombuffer(msg["qvel"], dtype=np.float64).copy()
    env.set_state({"qpos": qpos, "qvel": qvel})
    return {"type": RESP_OK}


def _handle_teacher_step(env: SO101Env, teacher: PickTeacher, teacher_mode: bool) -> dict:
    """Run one teacher step: extract obs, call teacher, step env."""
    if not teacher_mode:
        return {"type": RESP_ERROR, "message": "teacher_step requires teacher_mode=True (send configure first)"}

    obs = env._extract_obs()  # noqa: SLF001
    action, phase_id, done = teacher.step(obs, env.mujoco_model, env.mujoco_data)

    # Step the environment with the teacher action
    env.step(action)

    return {
        "type": RESP_TEACHER_STEP_OK,
        "action": ndarray_to_bytes(action),
        "phase_id": phase_id,
        "done": done,
        # Include observation data for client-side recording
        "qpos": ndarray_to_bytes(obs["qpos"]),
        "qvel": ndarray_to_bytes(obs["qvel"]),
        "ee_pose": ndarray_to_bytes(obs["ee_pose"]),
        "object_pose": ndarray_to_bytes(obs["object_pose"]),
        "joint_pos": ndarray_to_bytes(obs["joint_pos"]),
        "gripper": float(obs["gripper"]),
        "rgb_scene": ndarray_to_bytes(obs["rgb_scene"]),
        "rgb_wrist": ndarray_to_bytes(obs["rgb_wrist"]),
    }


def _handle_configure(msg: dict) -> dict:
    """Handle runtime configuration changes.

    Currently supports:
        teacher_mode (bool): Enable/disable teacher stepping.
    """
    # teacher_mode is tracked by the server main loop, not here.
    # This handler just acknowledges the configure command.
    logger.info("Configure: %s", {k: v for k, v in msg.items() if k != "type"})
    return {"type": RESP_OK}


def _handle_set_hint(msg: dict) -> dict:
    """Acknowledge a hint update delivered over CommandRPC.

    Returns RESP_OK with the parsed hint fields so the server main loop
    can store the latest hint from a single source of truth.
    """
    hint = {
        "ts_ms": msg.get("ts_ms"),
        "target_handle": msg.get("target_handle"),
        "bbox_xywh": msg.get("bbox_xywh"),
        "confidence": msg.get("confidence"),
        "tracker_ok": msg.get("tracker_ok"),
    }
    logger.debug("Set hint: %s", hint)
    return {"type": RESP_OK, "hint": hint}
