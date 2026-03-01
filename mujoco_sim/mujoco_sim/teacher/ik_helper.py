"""Damped least-squares IK: desired EE position → 5 arm joint angles.

Two solvers:
- ``solve_ik``: position-only (3D target, 3×5 Jacobian).
- ``solve_ik_with_orientation``: position + orientation (6D weighted cost, 6×5 Jacobian).
  Orientation cost penalises the gripperframe Z-axis deviating from a target Z-axis
  (typically world -Z for "gripper down"). Since the arm has only 5 DOF, full 6D
  control is impossible; the orientation term is weighted lower so position dominates.
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
    ori_weight: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-3,
    damping: float = 1e-2,
) -> np.ndarray:
    """Solve position + orientation IK using a two-phase approach.

    Phase 1: Position-only convergence (same as ``solve_ik``).
    Phase 2: Joint-space orientation refinement — nudge the null-space of the
    position Jacobian to improve orientation without degrading position.

    The 5-DOF SO-101 cannot fully control 6D pose, so orientation is best-effort.
    This two-phase approach guarantees position convergence first, then improves
    orientation as much as the kinematic chain allows.

    Operates on a *copy* of ``data`` to avoid mutating live sim state.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (will NOT be mutated).
        target_pos: (3,) desired EE position in world frame.
        target_rot: (3,3) desired EE rotation matrix in world frame.
        ee_site_id: MuJoCo site id for the end-effector (gripperframe).
        arm_joint_ids: List of 5 arm joint IDs.
        pos_weight: Weight for position error term.
        ori_weight: Weight for orientation error term (used in phase 2).
        max_iters: Maximum solver iterations (split between phases).
        tol: Position error tolerance (metres).
        damping: Damping coefficient λ.

    Returns:
        (5,) arm joint angles.
    """
    # Phase 1: position-only convergence
    pos_iters = max(max_iters * 2 // 3, 50)
    joints = solve_ik(
        model,
        data,
        target_pos,
        ee_site_id,
        arm_joint_ids,
        max_iters=pos_iters,
        tol=tol,
        damping=damping,
    )

    # Phase 2: orientation refinement with position maintenance
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qvel[:] = data.qvel[:]

    # Map joint IDs → qpos / Jacobian-column indices
    qpos_idx = [int(model.jnt_qposadr[jid]) for jid in arm_joint_ids]
    dof_idx = [int(model.jnt_dofadr[jid]) for jid in arm_joint_ids]

    for i, jid in enumerate(arm_joint_ids):
        d.qpos[qpos_idx[i]] = joints[i]
    mujoco.mj_forward(model, d)

    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    refine_iters = max_iters - pos_iters

    for _ in range(refine_iters):
        ee_pos = d.site_xpos[ee_site_id].copy()
        pos_err = target_pos - ee_pos
        ori_err = _orientation_error(d.site_xmat[ee_site_id], target_rot)

        # Combined error with strong position weight to prevent drift
        err = np.concatenate([pos_err * pos_weight, ori_err * ori_weight])
        if np.linalg.norm(err) < tol * 0.1:
            break

        mujoco.mj_jacSite(model, d, jac_pos, jac_rot, ee_site_id)

        Jp = jac_pos[:, dof_idx] * pos_weight
        Jr = jac_rot[:, dof_idx] * ori_weight
        J = np.vstack([Jp, Jr])

        JJT = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)

        # Limit step size to avoid overshooting
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 0.05:
            dq = dq * 0.05 / dq_norm

        for i, jid in enumerate(arm_joint_ids):
            d.qpos[qpos_idx[i]] += dq[i]
            lo, hi = model.jnt_range[jid]
            d.qpos[qpos_idx[i]] = np.clip(d.qpos[qpos_idx[i]], lo, hi)

        mujoco.mj_forward(model, d)

        # Abort refinement if position drifted too much
        new_pos_err = np.linalg.norm(target_pos - d.site_xpos[ee_site_id])
        if new_pos_err > tol * 5:
            # Revert to position-only solution
            return joints

    return np.array([d.qpos[qpos_idx[i]] for i in range(len(arm_joint_ids))], dtype=np.float64)
