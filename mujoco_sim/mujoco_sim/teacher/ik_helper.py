"""Damped least-squares IK: desired EE position → 5 arm joint angles.

Two solvers:
- ``solve_ik``: position-only (3D target, 3×5 Jacobian).
- ``solve_ik_with_orientation``: position + orientation (6D weighted cost, 6×5 Jacobian).
  Single-phase coupled solver: position and orientation are optimised simultaneously
  from iteration 1 using a combined ``[pos_weight * Jp; ori_weight * Jr]`` Jacobian.
  Position weight dominates so position converges first naturally, while orientation
  biases the solver toward better joint configurations from the start — avoiding
  the local-minimum trap of a two-phase approach where position locks in first.
  Since the arm has only 5 DOF, full 6D control is impossible; the orientation
  term is weighted lower so position dominates.
"""

from __future__ import annotations

import mujoco
import numpy as np


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    ee_site_id: int,
    arm_joint_ids: list[int],
    *,
    max_iters: int = 100,
    tol: float = 1e-3,
    damping: float = 1e-2,
) -> np.ndarray:
    """Solve position-only IK for the arm joints using damped least-squares.

    Operates on a *copy* of ``data`` to avoid mutating live sim state.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (will NOT be mutated — we work on a copy).
        target_pos: (3,) desired EE position in world frame.
        ee_site_id: MuJoCo site id for the end-effector (gripperframe).
        arm_joint_ids: List of 5 arm joint IDs (excludes gripper).
        max_iters: Maximum solver iterations.
        tol: Position error tolerance (metres).
        damping: Damping coefficient (λ² in (JᵀJ + λ²I)⁻¹Jᵀe).

    Returns:
        (5,) arm joint angles that place the EE near ``target_pos``.
    """
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qvel[:] = data.qvel[:]
    mujoco.mj_forward(model, d)

    # Map joint IDs → qpos / Jacobian-column indices (correct for any MJCF layout)
    qpos_idx = [int(model.jnt_qposadr[jid]) for jid in arm_joint_ids]
    dof_idx = [int(model.jnt_dofadr[jid]) for jid in arm_joint_ids]

    jac_pos = np.zeros((3, model.nv))

    for _ in range(max_iters):
        ee_pos = d.site_xpos[ee_site_id].copy()
        err = target_pos - ee_pos

        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacSite(model, d, jac_pos, None, ee_site_id)

        J = jac_pos[:, dof_idx]  # (3, n_joints)

        # Damped least-squares: dq = Jᵀ(JJᵀ + λ²I)⁻¹ e
        JJT = J @ J.T + damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)

        for i, jid in enumerate(arm_joint_ids):
            d.qpos[qpos_idx[i]] += dq[i]
            # Clamp to joint limits
            lo, hi = model.jnt_range[jid]
            d.qpos[qpos_idx[i]] = np.clip(d.qpos[qpos_idx[i]], lo, hi)

        mujoco.mj_forward(model, d)

    return np.array([d.qpos[qpos_idx[i]] for i in range(len(arm_joint_ids))], dtype=np.float64)


def _orientation_error(site_xmat: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    """Compute a 3D Z-axis orientation error between current and target rotations.

    Only penalises the gripperframe Z-axis (approach direction) deviating from
    the target Z-axis.  The cross-product output is a 3D angular-error vector
    compatible with the 3×n rotational Jacobian, so the refinement loop structure
    is unchanged.  For a 5-DOF arm this avoids wasting IK effort on axes it
    cannot fully control.
    """
    R_cur = site_xmat.reshape(3, 3)
    z_cur = R_cur[:, 2]
    z_tgt = target_rot[:, 2]
    return np.cross(z_cur, z_tgt)


def solve_ik_with_orientation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    ee_site_id: int,
    arm_joint_ids: list[int],
    *,
    pos_weight: float = 1.0,
    ori_weight: float = 0.3,
    max_iters: int = 200,
    tol: float = 1e-3,
    damping: float = 1e-2,
) -> np.ndarray:
    """Solve position + orientation IK using a single-phase coupled solver.

    Position and orientation are optimised simultaneously from iteration 1
    using a combined ``[pos_weight * Jp; ori_weight * Jr]`` 6×5 Jacobian.
    Position weight dominates, so position converges first naturally, while
    the orientation term biases the solver toward joint configurations that
    also satisfy the target Z-axis alignment — avoiding the local-minimum
    trap of a sequential approach.

    The 5-DOF SO-101 cannot fully control 6D pose, so orientation is
    best-effort.  The full iteration budget is available to the coupled
    solver (vs. the old two-phase split that gave orientation only 1/3 of
    the iterations).

    Operates on a *copy* of ``data`` to avoid mutating live sim state.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (will NOT be mutated).
        target_pos: (3,) desired EE position in world frame.
        target_rot: (3,3) desired EE rotation matrix in world frame.
        ee_site_id: MuJoCo site id for the end-effector (gripperframe).
        arm_joint_ids: List of 5 arm joint IDs.
        pos_weight: Weight for position error term.
        ori_weight: Weight for orientation error term.
        max_iters: Maximum solver iterations.
        tol: Position error tolerance (metres).
        damping: Damping coefficient λ.

    Returns:
        (5,) arm joint angles.
    """
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qvel[:] = data.qvel[:]
    mujoco.mj_forward(model, d)

    qpos_idx = [int(model.jnt_qposadr[jid]) for jid in arm_joint_ids]
    dof_idx = [int(model.jnt_dofadr[jid]) for jid in arm_joint_ids]

    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    step_limit = 0.1

    for _ in range(max_iters):
        ee_pos = d.site_xpos[ee_site_id].copy()
        pos_err = target_pos - ee_pos
        ori_err = _orientation_error(d.site_xmat[ee_site_id], target_rot)

        if np.linalg.norm(pos_err) < tol and np.linalg.norm(ori_err) < 0.01:
            break

        err = np.concatenate([pos_err * pos_weight, ori_err * ori_weight])

        mujoco.mj_jacSite(model, d, jac_pos, jac_rot, ee_site_id)

        Jp = jac_pos[:, dof_idx] * pos_weight
        Jr = jac_rot[:, dof_idx] * ori_weight
        J = np.vstack([Jp, Jr])

        JJT = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)

        dq_norm = np.linalg.norm(dq)
        if dq_norm > step_limit:
            dq = dq * step_limit / dq_norm

        for i, jid in enumerate(arm_joint_ids):
            d.qpos[qpos_idx[i]] += dq[i]
            lo, hi = model.jnt_range[jid]
            d.qpos[qpos_idx[i]] = np.clip(d.qpos[qpos_idx[i]], lo, hi)

        mujoco.mj_forward(model, d)

    return np.array([d.qpos[qpos_idx[i]] for i in range(len(arm_joint_ids))], dtype=np.float64)
