"""Trajectory-planned PICK teacher using privileged sim state.

Mirrors the PICK skill phase sequence from ``configs/skills/pick/default.mmd``.
On the first ``step()`` call, pre-computes the full trajectory (grasp evaluation →
keyframes → IK waypoints → jerk-limited ruckig segments).  Each subsequent ``step()``
samples the trajectory at ``t = step_count * dt``.

Usage::

    teacher = PickTeacher()
    teacher.reset()
    action, phase_id, done = teacher.step(obs, model, data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_sim.constants import (
    PHASE_DONE,
    PHASE_IDLE,
)
from mujoco_sim.scene_info import (
    DEFAULT_CUBE_FACE_CONTACT_SPAN,
    DEFAULT_FACE_STANDOFF,
    DEFAULT_GRASP_MAX_CONE_DEG,
    DEFAULT_GRASP_N_CANDIDATES,
    DEFAULT_IK_MAX_ITERS,
    DEFAULT_IK_ORI_WEIGHT,
    DEFAULT_IK_POS_TOL,
    DEFAULT_IK_POS_WEIGHT,
    DEFAULT_IK_TOL,
    DEFAULT_LIFT_HEIGHT,
    DEFAULT_ORI_TOL_DEG,
    DEFAULT_PREGRASP_STANDOFF,
    EE_SITE_NAME,
    SceneInfo,
)
from mujoco_sim.teacher.grasp_planner import evaluate_grasps
from mujoco_sim.teacher.keyframe_planner import plan_pick_keyframes
from mujoco_sim.teacher.trajectory import JointLimits, TrajectoryPlan, plan_trajectory
from mujoco_sim.teacher.waypoint_generator import generate_joint_waypoints

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Configuration for the trajectory-planned pick teacher."""

    # Standoff distance along approach direction for pregrasp
    pregrasp_height_offset: float = DEFAULT_PREGRASP_STANDOFF

    # Lift target height above grasp contact point
    lift_height: float = DEFAULT_LIFT_HEIGHT

    # Grasp planner orientation tolerance
    ori_tol_deg: float = DEFAULT_ORI_TOL_DEG

    # Grasp sampling (random candidates on cube side faces)
    grasp_n_candidates: int = DEFAULT_GRASP_N_CANDIDATES
    grasp_max_cone_deg: float = DEFAULT_GRASP_MAX_CONE_DEG
    grasp_face_contact_span: float = DEFAULT_CUBE_FACE_CONTACT_SPAN
    grasp_face_standoff: float = DEFAULT_FACE_STANDOFF

    # Trajectory limits (per-joint)
    max_velocity: list[float] | None = None  # defaults in JointLimits
    max_acceleration: list[float] | None = None
    max_jerk: list[float] | None = None

    # IK parameters
    ik_pos_weight: float = DEFAULT_IK_POS_WEIGHT
    ik_ori_weight: float = DEFAULT_IK_ORI_WEIGHT
    ik_max_iters: int = DEFAULT_IK_MAX_ITERS
    ik_tol: float = DEFAULT_IK_TOL
    ik_pos_tol: float = DEFAULT_IK_POS_TOL


class PickTeacher:
    """Trajectory-planned PICK policy using ground-truth sim state.

    Phase sequence recorded in trajectory segments::

        IDLE → MOVE_PREGRASP → EXECUTE_APPROACH → CLOSE_GRIPPER → LIFT → DONE

    SELECT_GRASP, PLAN_APPROACH, and VISUAL_ALIGN are folded into the planning
    step (instantaneous). VERIFY_GRASP is implicit in the gripper-close segment.

    The ``step()`` interface is identical to the original reactive teacher, so
    the runner, server, and recording pipeline remain untouched.
    """

    def __init__(self, config: TeacherConfig | None = None) -> None:
        self._config = config or TeacherConfig()
        self._phase = PHASE_IDLE
        self._plan: TrajectoryPlan | None = None
        self._t: float = 0.0
        self._step_count: int = 0
        self._dt: float | None = None

        # Cache IDs + scene info on first step (need model)
        self._ee_site_id: int | None = None
        self._arm_joint_ids: list[int] | None = None
        self._scene_info: SceneInfo | None = None

    def reset(self) -> None:
        """Reset teacher state for a new episode."""
        self._phase = PHASE_IDLE
        self._plan = None
        self._t = 0.0
        self._step_count = 0

    @property
    def phase(self) -> int:
        """Current phase ID."""
        return self._phase

    @property
    def done(self) -> bool:
        """Whether the pick task is complete."""
        return self._phase == PHASE_DONE

    def _ensure_ids(self, model: mujoco.MjModel) -> None:
        """Cache site/joint IDs and scene info on first call."""
        if self._ee_site_id is None:
            self._ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
            self._arm_joint_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
            ]
            self._scene_info = SceneInfo.from_model(model)

    def step(
        self,
        obs: dict,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[np.ndarray, int, bool]:
        """Compute one teacher action from the current observation.

        On the first call, plans the full trajectory. Subsequent calls sample
        at ``t = step_count * dt``.

        Args:
            obs: Observation dict from ``SO101Env`` with keys:
                ``ee_pose`` (7,), ``object_pose`` (7,), ``gripper`` (float), ``joint_pos`` (6,).
            model: MuJoCo model (for IK on first call).
            data: MuJoCo data (for IK on first call — will NOT be mutated).

        Returns:
            (action, phase_id, done) where action is (6,) joint-position targets.
        """
        self._ensure_ids(model)

        if self._plan is None:
            self._plan = self._build_plan(obs, model, data)
            self._t = 0.0
            self._step_count = 0
            # Infer dt from model timestep and substep count
            # SO101Env uses: substeps = round(1/(control_freq * model.opt.timestep))
            # So dt = substeps * model.opt.timestep = 1/control_freq
            # Default: 20 Hz → dt=0.05s
            self._dt = model.opt.timestep * max(1, round(1.0 / (20.0 * model.opt.timestep)))

        arm_joints, gripper, phase_id = self._plan.sample(self._t)
        action = np.concatenate([arm_joints, [gripper]])

        self._phase = phase_id
        self._step_count += 1
        self._t = self._step_count * self._dt

        done = self._t >= self._plan.total_duration
        if done:
            self._phase = PHASE_DONE

        return action, self._phase, done

    def _build_plan(
        self,
        obs: dict,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> TrajectoryPlan:
        """Build the full trajectory plan from current state."""
        cfg = self._config
        cube_pos = obs["object_pose"][:3]
        cube_quat = obs["object_pose"][3:]
        home_joints = obs["joint_pos"][:6].copy()

        # Step 1: Evaluate grasp candidates (enumerate → filter → score → best)
        best = evaluate_grasps(
            cube_pos=cube_pos,
            cube_quat=cube_quat,
            cube_half_sizes=self._scene_info.green_cube_half_sizes,
            model=model,
            data=data,
            ee_site_id=self._ee_site_id,
            arm_joint_ids=self._arm_joint_ids,
            seed_joints=home_joints[:5],
            standoff=cfg.pregrasp_height_offset,
            z_lift=cfg.lift_height,
            table_z=self._scene_info.table_z,
            n_candidates=cfg.grasp_n_candidates,
            max_cone_deg=cfg.grasp_max_cone_deg,
            face_contact_span=cfg.grasp_face_contact_span,
            face_standoff=cfg.grasp_face_standoff,
            pos_tol=cfg.ik_pos_tol,
            ori_tol_deg=cfg.ori_tol_deg,
        )

        logger.info(
            "Selected grasp: face=%s yaw=%d score=%.3f (pos_err=%.4f m, ori_err=%.1f°)",
            best.grasp.face_label,
            best.grasp.yaw_variant,
            best.score,
            best.ik_pos_err,
            best.ori_err_deg,
        )

        # Step 2: Cartesian keyframes from grasp pose
        keyframes = plan_pick_keyframes(
            home_joints=home_joints,
            grasp_pose=best.grasp,
            ee_site_id=self._ee_site_id,
            model=model,
            data=data,
            standoff=cfg.pregrasp_height_offset,
            z_lift=cfg.lift_height,
        )

        logger.info(
            "Planned %d keyframes: %s",
            len(keyframes),
            [kf.label for kf in keyframes],
        )

        # Step 3: IK → joint waypoints
        waypoints = generate_joint_waypoints(
            keyframes=keyframes,
            model=model,
            data=data,
            ee_site_id=self._ee_site_id,
            arm_joint_ids=self._arm_joint_ids,
            seed_joints=home_joints[:5],
            pos_weight=cfg.ik_pos_weight,
            ori_weight=cfg.ik_ori_weight,
            max_iters=cfg.ik_max_iters,
            tol=cfg.ik_tol,
            pos_tol=cfg.ik_pos_tol,
        )

        logger.info(
            "Solved IK for %d waypoints: %s",
            len(waypoints),
            [(wp.label, wp.arm_joints.round(3).tolist()) for wp in waypoints],
        )

        # Step 4: Jerk-limited trajectory
        limits = JointLimits(
            max_velocity=cfg.max_velocity or JointLimits().max_velocity,
            max_acceleration=cfg.max_acceleration or JointLimits().max_acceleration,
            max_jerk=cfg.max_jerk or JointLimits().max_jerk,
        )
        plan = plan_trajectory(waypoints, limits)

        logger.info(
            "Trajectory planned: %d segments, %.2f s total — %s",
            len(plan.segments),
            plan.total_duration,
            [(s.label, f"{s.duration:.2f}s") for s in plan.segments],
        )

        return plan
