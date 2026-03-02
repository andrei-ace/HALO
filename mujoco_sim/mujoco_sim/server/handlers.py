"""Command dispatch for CommandRPC (REQ/REP) messages.

Handles: step, reset, get_state, set_state, start_pick, configure, set_hint, shutdown.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Callable

import numpy as np

from mujoco_sim.server.protocol import (
    CMD_CONFIGURE,
    CMD_GET_STATE,
    CMD_RESET,
    CMD_SET_HINT,
    CMD_SET_STATE,
    CMD_SHUTDOWN,
    CMD_START_PICK,
    CMD_STEP,
    RESP_ERROR,
    RESP_OK,
    RESP_RESET_OK,
    RESP_START_PICK_ERROR,
    RESP_START_PICK_OK,
    RESP_STATE,
    RESP_STEP_OK,
    ndarray_to_bytes,
)

if TYPE_CHECKING:
    from mujoco_sim.env import SO101Env

logger = logging.getLogger(__name__)
_TARGET_SUFFIX_RE = re.compile(r"_\d+$")


def _resolve_target_body(
    *,
    requested_body: str,
    known_bodies: tuple[str, ...],
    name_to_body_id: Callable[[str], int],
) -> tuple[str, int]:
    """Resolve a requested target body name to an actual MuJoCo body.

    Resolution order:
    1) Exact body name
    2) Trailing numeric suffix stripped (e.g. ``red_cube_01`` -> ``red_cube``)
    """
    candidates = [requested_body]
    stripped = _TARGET_SUFFIX_RE.sub("", requested_body)
    if stripped != requested_body:
        candidates.append(stripped)

    for candidate in candidates:
        body_id = name_to_body_id(candidate)
        if body_id != -1:
            return candidate, body_id

    known = ", ".join(repr(name) for name in known_bodies)
    tried = ", ".join(repr(name) for name in candidates)
    raise KeyError(f"Unknown body: {requested_body!r}. Tried: [{tried}]. Known: [{known}]")


def dispatch_command(
    msg: dict,
    env: SO101Env,
    server_state: ServerState | None = None,
) -> tuple[dict, bool]:
    """Dispatch a CommandRPC command and return (response_dict, should_shutdown).

    Args:
        msg: Decoded msgpack command message.
        env: SO101Env instance.
        server_state: Mutable server state for trajectory tracking.

    Returns:
        (response_dict, should_shutdown) tuple.
    """
    cmd = msg.get("type")

    if cmd == CMD_STEP:
        return _handle_step(msg, env), False

    if cmd == CMD_RESET:
        return _handle_reset(msg, env, server_state), False

    if cmd == CMD_GET_STATE:
        return _handle_get_state(env), False

    if cmd == CMD_SET_STATE:
        return _handle_set_state(msg, env), False

    if cmd == CMD_START_PICK:
        return _handle_start_pick(msg, env, server_state), False

    if cmd == CMD_CONFIGURE:
        return _handle_configure(msg), False

    if cmd == CMD_SET_HINT:
        return _handle_set_hint(msg), False

    if cmd == CMD_SHUTDOWN:
        logger.info("Shutdown command received")
        return {"type": RESP_OK}, True

    return {"type": RESP_ERROR, "message": f"Unknown command: {cmd}"}, False


class ServerState:
    """Mutable state shared between main loop and handlers."""

    def __init__(self) -> None:
        self.trajectory = None  # TrajectoryPlan | None
        self.traj_start_time: float = 0.0
        self.phase_id: int = 0
        self.done: bool = False
        self.error: str | None = None

        # Fixed position target for hold mode (prevents gravity drift).
        # Set on reset (home_qpos) and after trajectory completes (final endpoint).
        self.hold_target: np.ndarray | None = None

        # Cached IDs (lazy init on first start_pick)
        self._ee_site_id: int | None = None
        self._arm_joint_ids: list[int] | None = None
        self._green_cube_body_id: int | None = None
        self._red_cube_body_id: int | None = None
        self._scene_info = None

    def reset_trajectory(self) -> None:
        """Clear active trajectory state."""
        self.trajectory = None
        self.traj_start_time = 0.0
        self.phase_id = 0
        self.done = False
        self.error = None


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


def _handle_reset(msg: dict, env: SO101Env, server_state: ServerState | None) -> dict:
    """Reset environment (and clear any active trajectory)."""
    seed = msg.get("seed")
    env.reset(seed=seed)
    if server_state is not None:
        server_state.reset_trajectory()
        server_state.hold_target = env.home_qpos.copy()
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


def _handle_start_pick(msg: dict, env: SO101Env, server_state: ServerState | None) -> dict:
    """Plan a pick trajectory for a specific target body and start executing it.

    Uses the same pipeline as PickTeacher: grasp planning → keyframes →
    waypoints → trajectory. The trajectory is stored in server_state and
    sampled by the autonomous physics loop.

    Requires ``target_body`` field in *msg* (e.g. ``"green_cube"``).
    """
    if server_state is None:
        return {"type": RESP_START_PICK_ERROR, "message": "server_state not available"}

    target_body = msg.get("target_body")
    if not target_body:
        return {"type": RESP_START_PICK_ERROR, "message": "start_pick requires 'target_body' field"}

    try:
        import time

        import mujoco as mj

        from mujoco_sim.scene_info import (
            EE_SITE_NAME,
            GREEN_CUBE_BODY_NAME,
            RED_CUBE_BODY_NAME,
            SceneInfo,
        )
        from mujoco_sim.teacher.grasp_planner import GraspPlanningFailure, evaluate_grasps
        from mujoco_sim.teacher.keyframe_planner import plan_pick_keyframes
        from mujoco_sim.teacher.pick_teacher import TeacherConfig
        from mujoco_sim.teacher.trajectory import JointLimits, plan_trajectory
        from mujoco_sim.teacher.waypoint_generator import IKFailure, generate_joint_waypoints

        model = env.mujoco_model
        data = env.mujoco_data
        config = TeacherConfig()

        # Lazy-init cached IDs and scene info
        if server_state._scene_info is None:
            server_state._scene_info = SceneInfo.from_model(model)
            server_state._ee_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
            server_state._arm_joint_ids = [
                mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
                for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
            ]

        scene_info = server_state._scene_info

        known_bodies = (GREEN_CUBE_BODY_NAME, RED_CUBE_BODY_NAME)
        try:
            resolved_target_body, target_body_id = _resolve_target_body(
                requested_body=target_body,
                known_bodies=known_bodies,
                name_to_body_id=lambda name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name),
            )
        except KeyError as exc:
            logger.warning("start_pick rejected unknown target body: %s", exc)
            return {"type": RESP_START_PICK_ERROR, "message": str(exc)}

        if resolved_target_body != target_body:
            logger.info(
                "start_pick: resolved requested target %r -> %r",
                target_body,
                resolved_target_body,
            )

        # Look up half-sizes for this body
        try:
            target_half_sizes = scene_info.half_sizes_for_body(resolved_target_body)
        except KeyError as exc:
            return {"type": RESP_START_PICK_ERROR, "message": str(exc)}

        # Read current state
        current_joints = np.array(data.qpos[:6], copy=True)
        cube_pos = data.xpos[target_body_id].copy()
        cube_quat = data.xquat[target_body_id].copy()

        # 1. Evaluate grasp candidates
        best = evaluate_grasps(
            cube_pos=cube_pos,
            cube_quat=cube_quat,
            cube_half_sizes=target_half_sizes,
            model=model,
            data=data,
            ee_site_id=server_state._ee_site_id,
            arm_joint_ids=server_state._arm_joint_ids,
            seed_joints=current_joints[:5],
            standoff=config.pregrasp_height_offset,
            z_lift=config.lift_height,
            table_z=scene_info.table_z,
            n_candidates=config.grasp_n_candidates,
            max_cone_deg=config.grasp_max_cone_deg,
            face_contact_span=config.grasp_face_contact_span,
            face_standoff=config.grasp_face_standoff,
            pos_tol=config.ik_pos_tol,
            ori_tol_deg=config.ori_tol_deg,
        )

        # 2. Cartesian keyframes from grasp pose
        keyframes = plan_pick_keyframes(
            home_joints=current_joints,
            grasp_pose=best.grasp,
            ee_site_id=server_state._ee_site_id,
            model=model,
            data=data,
            standoff=config.pregrasp_height_offset,
            z_lift=config.lift_height,
        )

        # 3. IK → joint waypoints
        waypoints = generate_joint_waypoints(
            keyframes=keyframes,
            model=model,
            data=data,
            ee_site_id=server_state._ee_site_id,
            arm_joint_ids=server_state._arm_joint_ids,
            seed_joints=current_joints[:5],
            pos_weight=config.ik_pos_weight,
            ori_weight=config.ik_ori_weight,
            max_iters=config.ik_max_iters,
            tol=config.ik_tol,
            pos_tol=config.ik_pos_tol,
        )

        # 4. Jerk-limited trajectory
        limits = JointLimits(
            max_velocity=config.max_velocity or JointLimits().max_velocity,
            max_acceleration=config.max_acceleration or JointLimits().max_acceleration,
            max_jerk=config.max_jerk or JointLimits().max_jerk,
        )
        trajectory = plan_trajectory(waypoints, limits)

        logger.info(
            "start_pick: planned %d segments, %.2f s total",
            len(trajectory.segments),
            trajectory.total_duration,
        )

        # Store trajectory for the physics loop
        server_state.trajectory = trajectory
        server_state.traj_start_time = time.monotonic()
        server_state.phase_id = 0
        server_state.done = False
        server_state.error = None

        return {
            "type": RESP_START_PICK_OK,
            "duration": trajectory.total_duration,
            "target_body": resolved_target_body,
            "target_body_requested": target_body,
            "target_body_resolved": resolved_target_body,
        }

    except (GraspPlanningFailure, IKFailure) as exc:
        logger.warning("start_pick planning failed: %s", exc)
        server_state.error = str(exc)
        return {
            "type": RESP_START_PICK_ERROR,
            "message": str(exc),
        }
    except Exception as exc:
        logger.exception("start_pick unexpected error")
        server_state.error = str(exc)
        return {
            "type": RESP_START_PICK_ERROR,
            "message": str(exc),
        }


def _handle_configure(msg: dict) -> dict:
    """Handle runtime configuration changes."""
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
